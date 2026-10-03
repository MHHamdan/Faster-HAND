"""
Compute WER and CER metrics for HAND model
Properly handles multi-GPU trained models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hand.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from hand.models.experimental.complete_hand_model import create_hand_model
import pickle
import os
import argparse
from tqdm import tqdm
import editdistance


def load_model_from_checkpoint(checkpoint_path, model_params, device):
    """
    Load model from checkpoint, handling both single-GPU and multi-GPU checkpoints

    Args:
        checkpoint_path: path to checkpoint file
        model_params: model parameters dictionary
        device: target device

    Returns:
        model: loaded model
        checkpoint: full checkpoint dict with metadata
    """
    # Create model
    model = create_hand_model(model_params)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load state dict
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model, checkpoint


def decode_predictions(logits, charset, sos_token=None, eos_token=None, pad_token=-1):
    """
    Decode model predictions to text

    Args:
        logits: model output logits [B, vocab_size, T]
        charset: character set list
        sos_token, eos_token: special token indices
        pad_token: padding token value

    Returns:
        texts: list of decoded strings
    """
    # Get predicted indices
    preds = torch.argmax(logits, dim=1)  # [B, T]

    texts = []
    for pred in preds:
        chars = []
        for idx in pred:
            idx_val = idx.item()

            # Skip special tokens
            if idx_val == pad_token:
                continue
            if sos_token is not None and idx_val == sos_token:
                continue
            if eos_token is not None and idx_val == eos_token:
                break

            # Map to character
            if 0 <= idx_val < len(charset):
                chars.append(charset[idx_val])

        texts.append(''.join(chars))

    return texts


def decode_labels(labels, charset, sos_token=None, eos_token=None, pad_token=-1):
    """
    Decode ground truth labels to text

    Args:
        labels: ground truth label indices [B, T]
        charset: character set list
        sos_token, eos_token: special token indices
        pad_token: padding token value

    Returns:
        texts: list of decoded strings
    """
    texts = []
    for label in labels:
        chars = []
        for idx in label:
            idx_val = idx.item()

            # Skip special tokens
            if idx_val == pad_token:
                continue
            if sos_token is not None and idx_val == sos_token:
                continue
            if eos_token is not None and idx_val == eos_token:
                break

            # Map to character
            if 0 <= idx_val < len(charset):
                chars.append(charset[idx_val])

        texts.append(''.join(chars))

    return texts


def compute_wer(pred_text, gt_text):
    """
    Compute Word Error Rate

    Args:
        pred_text: predicted text string
        gt_text: ground truth text string

    Returns:
        wer: word error rate
    """
    pred_words = pred_text.split()
    gt_words = gt_text.split()

    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    distance = editdistance.eval(pred_words, gt_words)
    wer = distance / len(gt_words)

    return wer


def compute_cer(pred_text, gt_text):
    """
    Compute Character Error Rate

    Args:
        pred_text: predicted text string
        gt_text: ground truth text string

    Returns:
        cer: character error rate
    """
    if len(gt_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0

    distance = editdistance.eval(pred_text, gt_text)
    cer = distance / len(gt_text)

    return cer


def evaluate_model(model, dataloader, charset, device, num_classes):
    """
    Evaluate model and compute WER/CER metrics

    Args:
        model: trained model
        dataloader: evaluation data loader
        charset: character set
        device: computation device
        num_classes: number of output classes

    Returns:
        metrics: dictionary with WER, CER, and other metrics
    """
    model.eval()

    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0

    # Special token indices (assuming SOS/EOS are at the end of vocab)
    sos_token = len(charset)  # vocab_size
    eos_token = len(charset) + 1  # vocab_size + 1
    pad_token = -1

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get images and labels
            if "img" in batch:
                images = batch["img"].to(device)
            elif "imgs" in batch:
                images = batch["imgs"].to(device)
            elif "image" in batch:
                images = batch["image"].to(device)
            else:
                continue

            if "label" in batch:
                labels = batch["label"].to(device)
            elif "labels" in batch:
                labels = batch["labels"].to(device)
            elif "text" in batch:
                labels = batch["text"].to(device)
            else:
                continue

            # Prepare line indices
            if "line_indices" in batch:
                line_indices = batch["line_indices"].to(device)
                index_in_lines = batch["index_in_lines"].to(device)
            else:
                B, T = labels.shape
                line_indices = torch.zeros(B, T, dtype=torch.long, device=device)
                index_in_lines = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

            # Forward pass
            outputs = model(
                images=images,
                tokens=labels,  # Teacher forcing for evaluation
                line_indices=line_indices,
                index_in_lines=index_in_lines,
                epoch=0
            )

            logits = outputs['logits']  # [B, vocab_size, T]

            # Decode predictions and ground truth
            pred_texts = decode_predictions(logits, charset, sos_token, eos_token, pad_token)
            gt_texts = decode_labels(labels, charset, sos_token, eos_token, pad_token)

            # Compute WER and CER for each sample
            for pred, gt in zip(pred_texts, gt_texts):
                wer = compute_wer(pred, gt)
                cer = compute_cer(pred, gt)

                total_wer += wer
                total_cer += cer
                num_samples += 1

    # Average metrics
    avg_wer = total_wer / num_samples if num_samples > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0

    metrics = {
        'WER': avg_wer * 100,  # Convert to percentage
        'CER': avg_cer * 100,
        'num_samples': num_samples,
        'num_classes': num_classes
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compute WER/CER metrics for HAND model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='READ_2016', help='Dataset name')
    parser.add_argument('--level', type=str, default='line',
                       choices=['line', 'page', 'double_page', 'triple_page'],
                       help='Curriculum level to evaluate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("HAND Model Evaluation - WER/CER Computation")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Level: {args.level}")
    print(f"Device: {device}")
    print()

    # Load charset
    level_to_dataset = {
        'line': f'{args.dataset}_line_synthetic',
        'page': f'{args.dataset}_page_sem',
        'double_page': f'{args.dataset}_double_page_sem',
        'triple_page': f'{args.dataset}_triple_page_sem'
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    formatted_root = os.path.join(script_dir, "../../../../formatted")
    dataset_path = os.path.join(formatted_root, level_to_dataset[args.level])
    labels_file = os.path.join(dataset_path, "labels.pkl")

    with open(labels_file, "rb") as f:
        data = pickle.load(f)
        charset = data["charset"]

    num_classes = len(charset) + 3  # charset + SOS + EOS + blank

    print(f"Charset size: {len(charset)}")
    print(f"Number of output classes: {num_classes}")
    print()

    # Model parameters (should match training)
    model_params = {
        "vocab_size": 89,
        "input_channels": 3,
        "enc_dim": 192,
        "dropout": 0.5,
        "pe_h_max": 500,
        "pe_w_max": 1000,
        "l_max": 15000,
        "dec_num_layers": 4,
        "dec_num_heads": 4,
        "dec_res_dropout": 0.1,
        "dec_pred_dropout": 0.1,
        "dec_att_dropout": 0.1,
        "dec_dim_feedforward": 192,
        "memory_size": 48,
        "attention_win": 100,
        "use_line_indices": True,
        "two_step_pos_enc_mode": "cat",
        "warmup_epochs": 150,
        "level_epochs": {
            'line': 150,
            'page': 200,
            'double_page': 250,
            'triple_page': 300
        },
        "lambda_layout": 0.1,
        "lambda_text": 0.6,
        "lambda_complexity": 0.1,
        "device": device,
        "batch_size": args.batch_size,
        "batch_decay": 0.8,
    }

    # Load model
    print("Loading model from checkpoint...")
    model, checkpoint_info = load_model_from_checkpoint(args.checkpoint, model_params, device)

    if 'epoch' in checkpoint_info:
        print(f"Checkpoint from epoch: {checkpoint_info['epoch']}")
    if 'best_valid_loss' in checkpoint_info:
        print(f"Best validation loss: {checkpoint_info['best_valid_loss']:.4f}")
    print()

    # Load dataset
    from train_hand_curriculum import get_dataset_params, load_dataset_for_level

    print("Loading evaluation dataset...")
    train_dataset, valid_dataset, _, collate_fn = load_dataset_for_level(
        args.dataset, args.level, model_params
    )

    # Create dataloader for validation set
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"Validation samples: {len(valid_dataset)}")
    print()

    # Evaluate
    print("Computing metrics...")
    metrics = evaluate_model(model, valid_loader, charset, device, num_classes)

    # Print results
    print()
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Number of samples evaluated: {metrics['num_samples']}")
    print(f"Number of output classes: {metrics['num_classes']}")
    print(f"Word Error Rate (WER): {metrics['WER']:.2f}%")
    print(f"Character Error Rate (CER): {metrics['CER']:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
