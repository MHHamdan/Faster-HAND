"""
Multi-GPU Training Script for HAND Model with Curriculum Learning
Implements distributed data parallel training across all available GPUs
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from hand.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from hand.models.experimental.complete_hand_model import create_hand_model
import numpy as np
import random
import os
import time
import pickle
from tqdm import tqdm
from collections import defaultdict


def set_seed(seed=0):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataset_params(dataset_name, level):
    """Get dataset parameters for a specific curriculum level"""
    level_to_dataset = {
        'line': f'{dataset_name}_line_synthetic',
        'page': f'{dataset_name}_page_sem',
        'double_page': f'{dataset_name}_double_page_sem',
        'triple_page': f'{dataset_name}_triple_page_sem'
    }

    dataset_path = f"../../../../formatted/{level_to_dataset[level]}"

    params = {
        "dataset_manager": OCRDatasetManager,
        "dataset_class": OCRDataset,
        "batch_size": 4,
        "num_workers": 4,
        "datasets": {
            dataset_name: dataset_path,
        },
        "train": {
            "name": f"{dataset_name}-train-{level}",
            "datasets": [(dataset_name, "train"), ],
        },
        "valid": {
            f"{dataset_name}-valid-{level}": [(dataset_name, "valid"), ],
        },
        "config": {
            "width_divisor": 8,
            "height_divisor": 32,
            "padding_value": 0,
            "padding_token": -1,
            "charset_mode": "seq2seq",
            "constraints": ["add_eot", "add_sot"],
            "preprocessings": [
                {
                    "type": "to_RGB",
                },
            ],
        }
    }

    return params


def load_dataset_for_level(dataset_name, level, model_params, rank):
    """Load and prepare dataset for specific curriculum level"""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Loading {level.upper()} level dataset")
        print(f"{'='*60}")

    dataset_params = get_dataset_params(dataset_name, level)

    # Load charset
    level_to_dataset = {
        'line': f'{dataset_name}_line_synthetic',
        'page': f'{dataset_name}_page_sem',
        'double_page': f'{dataset_name}_double_page_sem',
        'triple_page': f'{dataset_name}_triple_page_sem'
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    formatted_root = os.path.join(script_dir, "../../../../formatted")
    dataset_path = os.path.join(formatted_root, level_to_dataset[level])
    labels_file = os.path.join(dataset_path, "labels.pkl")

    with open(labels_file, "rb") as f:
        data = pickle.load(f)
        charset = data["charset"]
        if rank == 0:
            print(f"Charset size: {len(charset)}")

    dataset_params["charset"] = charset

    # Create dataset manager
    dataset_manager = OCRDatasetManager(dataset_params)
    dataset_manager.load_datasets()

    # Get datasets
    train_dataset = dataset_manager.train_dataset
    valid_dataset = dataset_manager.valid_datasets[list(dataset_manager.valid_datasets.keys())[0]]
    collate_fn = dataset_manager.my_collate_function

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Valid samples: {len(valid_dataset)}")

    return train_dataset, valid_dataset, charset, collate_fn


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all GPUs"""
    if world_size == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train_epoch(model, train_loader, optimizer, device, epoch, rank, world_size):
    """Train for one epoch"""
    model.train()
    epoch_losses = defaultdict(float)
    num_batches = 0

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        if "img" in batch:
            images = batch["img"].to(device)
        elif "imgs" in batch:
            images = batch["imgs"].to(device)
        elif "image" in batch:
            images = batch["image"].to(device)
        else:
            raise KeyError("No image key found in batch")

        if "label" in batch:
            tokens = batch["label"].to(device)
        elif "labels" in batch:
            tokens = batch["labels"].to(device)
        elif "text" in batch:
            tokens = batch["text"].to(device)
        else:
            raise KeyError("No label key found in batch")

        # Prepare line indices
        if "line_indices" in batch:
            line_indices = batch["line_indices"].to(device)
            index_in_lines = batch["index_in_lines"].to(device)
        else:
            B, T = tokens.shape
            line_indices = torch.zeros(B, T, dtype=torch.long, device=device)
            index_in_lines = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            images=images,
            tokens=tokens,
            line_indices=line_indices,
            index_in_lines=index_in_lines,
            epoch=epoch
        )

        # Compute loss
        loss, loss_dict = model.module.compute_loss(outputs, tokens)

        # Debug: Check for zero or NaN loss
        if rank == 0 and batch_idx == 0 and epoch % 10 == 0:
            valid_tokens = (tokens != -1).sum().item()
            total_tokens = tokens.numel()
            print(f"  [Debug] Batch 0: valid_tokens={valid_tokens}/{total_tokens}, loss={loss.item():.6e}")
            if loss.item() == 0.0:
                print(f"  [Debug] ⚠️  Zero loss! Tokens range: [{tokens.min().item()}, {tokens.max().item()}]")

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Accumulate losses (synchronize across GPUs)
        for k, v in loss_dict.items():
            v_tensor = torch.tensor(v, device=device)
            reduced_v = reduce_tensor(v_tensor, world_size)
            epoch_losses[k] += reduced_v.item()
        num_batches += 1

        # Update progress bar (only on rank 0)
        if rank == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'comp': loss_dict.get('complexity', 0)
            })

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= num_batches

    return dict(epoch_losses)


def validate(model, valid_loader, device, epoch, rank, world_size):
    """Validate the model"""
    model.eval()
    epoch_losses = defaultdict(float)
    num_batches = 0

    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(valid_loader, desc="Validation")
        else:
            pbar = valid_loader

        for batch in pbar:
            if "img" in batch:
                images = batch["img"].to(device)
            elif "imgs" in batch:
                images = batch["imgs"].to(device)
            elif "image" in batch:
                images = batch["image"].to(device)
            else:
                raise KeyError(f"No image key in batch")

            if "label" in batch:
                tokens = batch["label"].to(device)
            elif "labels" in batch:
                tokens = batch["labels"].to(device)
            elif "text" in batch:
                tokens = batch["text"].to(device)
            else:
                raise KeyError(f"No label key in batch")

            if "line_indices" in batch:
                line_indices = batch["line_indices"].to(device)
                index_in_lines = batch["index_in_lines"].to(device)
            else:
                B, T = tokens.shape
                line_indices = torch.zeros(B, T, dtype=torch.long, device=device)
                index_in_lines = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

            outputs = model(
                images=images,
                tokens=tokens,
                line_indices=line_indices,
                index_in_lines=index_in_lines,
                epoch=epoch
            )

            loss, loss_dict = model.module.compute_loss(outputs, tokens)

            # Synchronize losses across GPUs
            for k, v in loss_dict.items():
                v_tensor = torch.tensor(v, device=device)
                reduced_v = reduce_tensor(v_tensor, world_size)
                epoch_losses[k] += reduced_v.item()
            num_batches += 1

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= num_batches

    return dict(epoch_losses)


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint to resume from"""
    checkpoint_dir = f"{output_dir}/checkpoints"

    if not os.path.exists(checkpoint_dir):
        return None, None

    # Check for best model first
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    # Find all epoch checkpoints
    epoch_checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("epoch_") and fname.endswith(".pt"):
            try:
                epoch_num = int(fname.replace("epoch_", "").replace(".pt", ""))
                epoch_checkpoints.append((epoch_num, os.path.join(checkpoint_dir, fname)))
            except ValueError:
                continue

    # Prefer best_model.pt if it exists (it has complete state)
    # Otherwise use latest epoch checkpoint
    if os.path.exists(best_model_path):
        return best_model_path, None
    elif epoch_checkpoints:
        epoch_checkpoints.sort(reverse=True)
        latest_epoch, latest_path = epoch_checkpoints[0]
        return latest_path, latest_epoch

    return None, None


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, rank):
    """Load checkpoint and return start epoch and best loss"""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return 0, float('inf')

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"RESUMING from checkpoint: {checkpoint_path}")
        print(f"{'='*70}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect checkpoint format
    # New format: dict with 'model_state_dict', 'optimizer_state_dict', etc.
    # Old format: raw state_dict (just model weights)
    if 'model_state_dict' in checkpoint:
        # New format - complete checkpoint
        if rank == 0:
            print("✓ Complete checkpoint format detected")

        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            if rank == 0:
                print(f"❌ Error loading checkpoint: {e}")
                print("⚠️  Checkpoint may be incompatible (e.g., different vocab_size)")
                print("⚠️  Starting training from scratch instead")
            return 0, float('inf')

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0:
                print("✓ Loaded optimizer state")

        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if rank == 0:
                print("✓ Loaded scheduler state")

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))

        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")
            print(f"Best validation loss: {best_valid_loss:.6e}")
            if 'curriculum_level' in checkpoint:
                print(f"Previous curriculum level: {checkpoint['curriculum_level']}")
    else:
        # Old format - only model weights
        if rank == 0:
            print("⚠️  Old checkpoint format (model weights only)")

        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            if rank == 0:
                print(f"❌ Error loading checkpoint: {e}")
                print("⚠️  Checkpoint may be incompatible (e.g., different vocab_size)")
                print("⚠️  Starting training from scratch instead")
            return 0, float('inf')

        # Extract epoch from filename if possible
        import re
        match = re.search(r'epoch_(\d+)', checkpoint_path)
        if match:
            start_epoch = int(match.group(1)) + 1
        else:
            start_epoch = 0

        best_valid_loss = float('inf')

        if rank == 0:
            print(f"⚠️  Optimizer and scheduler states not available - will use fresh states")
            print(f"Estimated starting epoch: {start_epoch}")

    if rank == 0:
        print(f"{'='*70}\n")

    return start_epoch, best_valid_loss


def train_worker(rank, world_size):
    """Main training worker for each GPU"""
    # Setup distributed training
    setup_distributed(rank, world_size)

    # Set seed
    set_seed(0 + rank)  # Different seed per process for data augmentation diversity

    # Configuration
    dataset_name = "READ_2016"
    max_epochs = 900
    device = torch.device(f'cuda:{rank}')
    output_dir = "outputs/hand_curriculum_multigpu"

    if rank == 0:
        print("="*70)
        print("HAND Model Training with Multi-GPU Curriculum Learning")
        print("="*70)
        print(f"World Size (# GPUs): {world_size}")
        print(f"Dataset: {dataset_name}")
        print(f"Max epochs: {max_epochs}")

    # Model parameters
    # vocab_size must accommodate ALL curriculum levels (max charset = 99)
    model_params = {
        "vocab_size": 99,  # Fixed: was 89, but page/double/triple levels need 99
        "input_channels": 3,
        "enc_dim": 192,
        "dropout": 0.5,

        # Encoder
        "pe_h_max": 500,
        "pe_w_max": 1000,

        # Decoder
        "l_max": 15000,
        "dec_num_layers": 4,
        "dec_num_heads": 4,
        "dec_res_dropout": 0.1,
        "dec_pred_dropout": 0.1,
        "dec_att_dropout": 0.1,
        "dec_dim_feedforward": 192,

        # HAND-specific
        "memory_size": 48,
        "attention_win": 100,
        "use_line_indices": True,
        "two_step_pos_enc_mode": "cat",

        # MSAP
        "warmup_epochs": 150,
        "level_epochs": {
            'line': 150,
            'page': 200,
            'double_page': 250,
            'triple_page': 300
        },

        # Loss weights
        "lambda_layout": 0.1,
        "lambda_text": 0.6,
        "lambda_complexity": 0.1,

        # Device
        "device": device,
        "batch_size": 2,  # Per GPU batch size (reduced for memory)
        "batch_decay": 0.8,
    }

    # Create model
    if rank == 0:
        print("\nCreating HAND model...")

    model = create_hand_model(model_params)
    model = model.to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Create output directories (only on rank 0)
    if rank == 0:
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)

    # Check for existing checkpoint to resume from (unless fresh start)
    fresh_start = os.environ.get('HAND_FRESH_START', '0') == '1'
    if fresh_start:
        if rank == 0:
            print("\n⚠️  Starting fresh training (checkpoints ignored)\n")
        checkpoint_path = None
        start_epoch = 0
        best_valid_loss = float('inf')
    else:
        checkpoint_path, checkpoint_epoch = find_latest_checkpoint(output_dir)
        start_epoch, best_valid_loss = load_checkpoint(
            checkpoint_path, model.module, optimizer, scheduler, device, rank
        )

        # Synchronize all processes after loading checkpoint
        if checkpoint_path is not None:
            dist.barrier()

    # Training loop with curriculum
    current_level = None
    train_loader = None
    valid_loader = None
    train_sampler = None
    valid_sampler = None

    for epoch in range(start_epoch, max_epochs):
        # Check curriculum level
        level_idx, level = model.module.curriculum_scheduler.get_current_level(epoch)

        # Load new dataset if level changed
        if level != current_level:
            current_level = level
            adaptive_batch_size = model.module.curriculum_scheduler.get_batch_size(level_idx)

            if rank == 0:
                print(f"\n{'='*70}")
                print(f"CURRICULUM SWITCH: {current_level.upper()}")
                print(f"Epoch {epoch} - Batch size per GPU: {adaptive_batch_size}")
                print(f"Effective batch size: {adaptive_batch_size * world_size}")
                print(f"{'='*70}")

            # Load dataset for current level
            train_dataset, valid_dataset, charset, collate_fn = load_dataset_for_level(
                dataset_name, current_level, model_params, rank
            )

            # Create distributed samplers
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=0
            )

            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=0
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=adaptive_batch_size,
                sampler=train_sampler,
                num_workers=2,  # Reduced per GPU
                pin_memory=True,
                collate_fn=collate_fn
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=adaptive_batch_size,
                sampler=valid_sampler,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn
            )

        # Set epoch for sampler (for proper shuffling)
        train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch, rank, world_size)

        # Validate
        valid_losses = validate(model, valid_loader, device, epoch, rank, world_size)

        # Print epoch summary (only on rank 0)
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{max_epochs} - Level: {current_level}")
            print(f"Train - Loss: {train_losses['total']:.6e}, Text: {train_losses['text']:.6e}, Comp: {train_losses['complexity']:.6e}")
            print(f"Valid - Loss: {valid_losses['total']:.6e}, Text: {valid_losses['text']:.6e}, Comp: {valid_losses['complexity']:.6e}")

            # Debug: Check for NaN or suspicious values
            if train_losses['total'] == 0.0 or valid_losses['total'] == 0.0:
                print("⚠️ WARNING: Zero loss detected! Possible model collapse or data issue.")

        # Save checkpoint (only on rank 0)
        if rank == 0:
            if valid_losses['total'] < best_valid_loss:
                best_valid_loss = valid_losses['total']

                # Save unwrapped model (without DDP wrapper)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # Use .module to get underlying model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'valid_losses': valid_losses,
                    'curriculum_level': current_level,
                    'best_valid_loss': best_valid_loss,
                }
                torch.save(checkpoint, f"{output_dir}/checkpoints/best_model.pt")
                print(f"✓ Saved best model (valid loss: {best_valid_loss:.6e})")

            # Periodic checkpoint (every 50 epochs)
            if (epoch + 1) % 50 == 0:
                periodic_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'valid_losses': valid_losses,
                    'curriculum_level': current_level,
                    'best_valid_loss': best_valid_loss,
                }
                torch.save(periodic_checkpoint, f"{output_dir}/checkpoints/epoch_{epoch+1}.pt")
                print(f"✓ Saved periodic checkpoint at epoch {epoch+1}")

        # Update scheduler
        scheduler.step()

        # Synchronize all processes
        dist.barrier()

    if rank == 0:
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best validation loss: {best_valid_loss:.4f}")
        print("="*70)

    # Cleanup
    cleanup_distributed()


def main():
    """Main entry point for multi-GPU training"""
    import argparse

    parser = argparse.ArgumentParser(description='HAND Multi-GPU Training')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Start training from scratch, ignoring existing checkpoints')
    args = parser.parse_args()

    # Store fresh_start flag in environment for worker processes
    if args.fresh_start:
        os.environ['HAND_FRESH_START'] = '1'
        print("⚠️  Fresh start mode: Existing checkpoints will be ignored")

    world_size = torch.cuda.device_count()

    if world_size < 2:
        print(f"Warning: Only {world_size} GPU(s) available. Consider using single-GPU training.")
        if world_size == 0:
            print("No GPUs available!")
            return

    print(f"Starting multi-GPU training with {world_size} GPUs")

    # Spawn processes for each GPU
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
