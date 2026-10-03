"""
Complete Training Script for HAND Model with Curriculum Learning
Implements full training pipeline with automatic dataset switching
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from hand.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from hand.models.experimental.complete_hand_model import create_hand_model
from hand.basic.transforms import aug_config
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


def get_dataset_params(dataset_name, level):
    """
    Get dataset parameters for a specific curriculum level

    Args:
        dataset_name: dataset name (e.g., "READ_2016")
        level: curriculum level (line, page, double_page, triple_page)

    Returns:
        params dict for dataset manager
    """
    # Map curriculum levels to formatted dataset names
    level_to_dataset = {
        'line': f'{dataset_name}_line_synthetic',
        'page': f'{dataset_name}_page_sem',
        'double_page': f'{dataset_name}_double_page_sem',
        'triple_page': f'{dataset_name}_triple_page_sem'
    }

    # Path relative to hand/OCR/document_OCR/hand/ directory
    dataset_path = f"../../../../formatted/{level_to_dataset[level]}"

    params = {
        "dataset_manager": OCRDatasetManager,
        "dataset_class": OCRDataset,
        "batch_size": 4,  # Will be overridden by curriculum batch size
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


def load_dataset_for_level(dataset_name, level, model_params):
    """
    Load and prepare dataset for specific curriculum level

    Args:
        dataset_name: dataset name
        level: curriculum level
        model_params: model parameters containing vocab info

    Returns:
        train_loader, valid_loader
    """
    print(f"\n{'='*60}")
    print(f"Loading {level.upper()} level dataset")
    print(f"{'='*60}")

    dataset_params = get_dataset_params(dataset_name, level)

    # Load charset from formatted dataset
    level_to_dataset = {
        'line': f'{dataset_name}_line_synthetic',
        'page': f'{dataset_name}_page_sem',
        'double_page': f'{dataset_name}_double_page_sem',
        'triple_page': f'{dataset_name}_triple_page_sem'
    }

    # Get absolute path to formatted datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    formatted_root = os.path.join(script_dir, "../../../../formatted")
    dataset_path = os.path.join(formatted_root, level_to_dataset[level])
    labels_file = os.path.join(dataset_path, "labels.pkl")

    with open(labels_file, "rb") as f:
        data = pickle.load(f)
        charset = data["charset"]
        print(f"Charset size: {len(charset)}")

    dataset_params["charset"] = charset

    # Create dataset manager
    dataset_manager = OCRDatasetManager(dataset_params)

    # Load datasets
    dataset_manager.load_datasets()

    # Get train and valid datasets
    train_dataset = dataset_manager.train_dataset
    valid_dataset = dataset_manager.valid_datasets[list(dataset_manager.valid_datasets.keys())[0]]
    collate_fn = dataset_manager.my_collate_function

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    return train_dataset, valid_dataset, charset, collate_fn


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    epoch_losses = defaultdict(float)
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Debug: print batch info on first iteration
        if batch_idx == 0 and epoch == 0:
            print(f"\nBatch keys: {batch.keys()}")
            print(f"Images shape: {batch['imgs'].shape if 'imgs' in batch else 'N/A'}")
            print(f"Labels shape: {batch['labels'].shape if 'labels' in batch else 'N/A'}")

        # Move batch to device
        # Check which key is used for images
        if "img" in batch:
            images = batch["img"].to(device)
        elif "imgs" in batch:
            images = batch["imgs"].to(device)
        elif "image" in batch:
            images = batch["image"].to(device)
        else:
            print(f"Available keys: {batch.keys()}")
            raise KeyError("No image key found in batch")

        # Check which key is used for labels
        if "label" in batch:
            tokens = batch["label"].to(device)
        elif "labels" in batch:
            tokens = batch["labels"].to(device)
        elif "text" in batch:
            tokens = batch["text"].to(device)
        else:
            raise KeyError("No label key found in batch")

        # Prepare line indices for hierarchical encoding
        if "line_indices" in batch:
            line_indices = batch["line_indices"].to(device)
            index_in_lines = batch["index_in_lines"].to(device)
        else:
            # Create dummy line indices for line-level data
            # All lines are line 0, positions are sequential
            B, T = tokens.shape
            line_indices = torch.zeros(B, T, dtype=torch.long, device=device)
            # Create sequential positions for each sample
            index_in_lines = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

            if batch_idx == 0 and epoch == 0:
                print(f"Tokens shape: {tokens.shape}, range: [{tokens.min()}, {tokens.max()}], unique: {tokens.unique().numel()}")
                print(f"Created line_indices shape: {line_indices.shape}, range: [{line_indices.min()}, {line_indices.max()}]")
                print(f"Created index_in_lines shape: {index_in_lines.shape}, range: [{index_in_lines.min()}, {index_in_lines.max()}]")
                print(f"Model vocab_size: {model.decoder.embedding.num_embeddings}")

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
        loss, loss_dict = model.compute_loss(outputs, tokens)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            epoch_losses[k] += v
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'comp': loss_dict.get('complexity', 0)
        })

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= num_batches

    return dict(epoch_losses)


def validate(model, valid_loader, device, epoch):
    """Validate the model"""
    model.eval()
    epoch_losses = defaultdict(float)
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):
            # Handle flexible batch keys
            if "img" in batch:
                images = batch["img"].to(device)
            elif "imgs" in batch:
                images = batch["imgs"].to(device)
            elif "image" in batch:
                images = batch["image"].to(device)
            else:
                raise KeyError(f"No image key in batch. Available: {batch.keys()}")

            if "label" in batch:
                tokens = batch["label"].to(device)
            elif "labels" in batch:
                tokens = batch["labels"].to(device)
            elif "text" in batch:
                tokens = batch["text"].to(device)
            else:
                raise KeyError(f"No label key in batch. Available: {batch.keys()}")

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

            loss, loss_dict = model.compute_loss(outputs, tokens)

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1

    for k in epoch_losses:
        epoch_losses[k] /= num_batches

    return dict(epoch_losses)


def train_hand_curriculum():
    """
    Main training function with automatic curriculum progression
    """
    # Set seed
    set_seed(0)

    # Configuration
    dataset_name = "READ_2016"
    max_epochs = 900  # Total epochs across all curriculum levels
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "outputs/hand_curriculum"

    print("="*70)
    print("HAND Model Training with Curriculum Learning")
    print("="*70)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Max epochs: {max_epochs}")

    # Model parameters
    model_params = {
        "vocab_size": 89,  # Will be updated from dataset
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
        "batch_size": 4,
        "batch_decay": 0.8,
    }

    # Create model
    print("\nCreating HAND model...")
    model = create_hand_model(model_params)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Create output directories
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    # Training loop with curriculum
    current_level = None
    train_loader = None
    valid_loader = None

    best_valid_loss = float('inf')

    for epoch in range(max_epochs):
        # Check curriculum level
        level_idx, level = model.curriculum_scheduler.get_current_level(epoch)

        # Load new dataset if level changed
        if level != current_level:
            current_level = level
            adaptive_batch_size = model.curriculum_scheduler.get_batch_size(level_idx)

            print(f"\n{'='*70}")
            print(f"CURRICULUM SWITCH: {current_level.upper()}")
            print(f"Epoch {epoch} - Batch size: {adaptive_batch_size}")
            print(f"{'='*70}")

            # Load dataset for current level
            train_dataset, valid_dataset, charset, collate_fn = load_dataset_for_level(
                dataset_name, current_level, model_params
            )

            # Update vocab size if needed
            if len(charset) + 3 != model_params["vocab_size"]:
                print(f"Note: Vocab size mismatch. Dataset has {len(charset)} chars, model expects {model_params['vocab_size']}")

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=adaptive_batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=adaptive_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn
            )

        # Train for one epoch
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        valid_losses = validate(model, valid_loader, device, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{max_epochs} - Level: {current_level}")
        print(f"Train - Loss: {train_losses['total']:.4f}, Text: {train_losses['text']:.4f}, Comp: {train_losses['complexity']:.4f}")
        print(f"Valid - Loss: {valid_losses['total']:.4f}, Text: {valid_losses['text']:.4f}, Comp: {valid_losses['complexity']:.4f}")

        # Save checkpoint
        if valid_losses['total'] < best_valid_loss:
            best_valid_loss = valid_losses['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'curriculum_level': current_level,
                'best_valid_loss': best_valid_loss,
            }
            torch.save(checkpoint, f"{output_dir}/checkpoints/best_model.pt")
            print(f"✓ Saved best model (valid loss: {best_valid_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"{output_dir}/checkpoints/epoch_{epoch+1}.pt")

        # Update scheduler
        scheduler.step()

    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    train_hand_curriculum()
