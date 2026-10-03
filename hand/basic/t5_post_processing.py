"""
T5 Post-Processing Module for OCR Error Correction

Based on the HAND paper methodology - uses T5 (smaller than mT5) for post-processing
to correct OCR errors in the predicted text.

The T5 model is fine-tuned on pairs of (corrupted OCR prediction, ground truth)
to learn to correct common OCR errors.
"""

import os
import re
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
    from transformers import TrainingArguments, Trainer
    from transformers import DataCollatorForSeq2Seq
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not available. T5 post-processing disabled.")


@dataclass
class T5PostProcessorConfig:
    """Configuration for T5 post-processing"""
    model_name: str = "t5-small"  # Use t5-small (60M params) instead of mT5
    max_source_length: int = 512
    max_target_length: int = 512
    batch_size: int = 8
    num_beams: int = 4
    early_stopping: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Layout tokens to preserve during post-processing
    layout_tokens: Optional[str] = None


class T5PostProcessor:
    """
    T5-based post-processor for OCR error correction.

    This module can:
    1. Load a pre-trained or fine-tuned T5 model
    2. Post-process OCR predictions to correct errors
    3. Preserve layout tokens while correcting text content

    Usage:
        # Initialize
        processor = T5PostProcessor(config)

        # For inference:
        corrected = processor.correct(ocr_prediction)

        # For batch inference:
        corrected_list = processor.correct_batch(ocr_predictions)
    """

    def __init__(self, config: Optional[T5PostProcessorConfig] = None, model_path: Optional[str] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for T5 post-processing")

        self.config = config or T5PostProcessorConfig()
        self.device = torch.device(self.config.device)

        # Load model
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned T5 model from {model_path}")
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        else:
            print(f"Loading pre-trained T5 model: {self.config.model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)

        self.model.to(self.device)
        self.model.eval()

        # Compile layout token pattern if provided
        self.layout_pattern = None
        if self.config.layout_tokens:
            escaped = re.escape(self.config.layout_tokens)
            self.layout_pattern = re.compile(f'[{escaped}]')

    def _extract_layout_info(self, text: str) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Extract layout tokens and their positions from text.
        Returns: (text_without_layout, list of (position, token) pairs)
        """
        if not self.layout_pattern:
            return text, []

        layout_info = []
        clean_text_parts = []
        current_pos = 0
        last_end = 0

        for match in self.layout_pattern.finditer(text):
            # Add text before this token
            clean_text_parts.append(text[last_end:match.start()])
            current_pos += match.start() - last_end
            # Record token position (in clean text) and the token itself
            layout_info.append((current_pos, match.group()))
            last_end = match.end()

        # Add remaining text
        clean_text_parts.append(text[last_end:])

        return ''.join(clean_text_parts), layout_info

    def _restore_layout_tokens(self, text: str, layout_info: List[Tuple[int, str]],
                                original_text: str) -> str:
        """
        Restore layout tokens to corrected text.
        Uses a heuristic approach to place tokens at appropriate positions.
        """
        if not layout_info:
            return text

        # Simple approach: try to place tokens at proportional positions
        result = list(text)
        original_len = len(original_text) - len(layout_info)  # Length without layout tokens
        new_len = len(text)

        # Sort layout info by position (descending) to insert from end
        layout_info_sorted = sorted(layout_info, key=lambda x: x[0], reverse=True)

        for orig_pos, token in layout_info_sorted:
            # Scale position proportionally
            if original_len > 0:
                new_pos = int(orig_pos * new_len / original_len)
            else:
                new_pos = 0
            new_pos = max(0, min(new_pos, len(result)))
            result.insert(new_pos, token)

        return ''.join(result)

    def correct(self, text: str, preserve_layout: bool = True) -> str:
        """
        Correct OCR errors in a single text string.

        Args:
            text: OCR prediction to correct
            preserve_layout: Whether to preserve layout tokens

        Returns:
            Corrected text
        """
        return self.correct_batch([text], preserve_layout)[0]

    def correct_batch(self, texts: List[str], preserve_layout: bool = True) -> List[str]:
        """
        Correct OCR errors in a batch of texts.

        Args:
            texts: List of OCR predictions
            preserve_layout: Whether to preserve layout tokens

        Returns:
            List of corrected texts
        """
        if not texts:
            return []

        # Extract layout info if needed
        layout_infos = []
        clean_texts = []
        for text in texts:
            if preserve_layout and self.layout_pattern:
                clean_text, layout_info = self._extract_layout_info(text)
                clean_texts.append(clean_text)
                layout_infos.append(layout_info)
            else:
                clean_texts.append(text)
                layout_infos.append([])

        # Prepare input for T5 (add task prefix)
        input_texts = [f"correct: {t}" for t in clean_texts]

        # Tokenize
        inputs = self.tokenizer(
            input_texts,
            max_length=self.config.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate corrections
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.config.max_target_length,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping,
                do_sample=False
            )

        # Decode outputs
        corrected_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Restore layout tokens if needed
        if preserve_layout:
            corrected_texts = [
                self._restore_layout_tokens(corrected, layout_info, original)
                for corrected, layout_info, original in zip(corrected_texts, layout_infos, texts)
            ]

        return corrected_texts


class T5FineTuner:
    """
    Fine-tune T5 for OCR error correction.

    Training data format:
        - Source: corrupted OCR text (with "correct: " prefix)
        - Target: ground truth text

    Usage:
        tuner = T5FineTuner(config)
        tuner.prepare_data(ocr_predictions, ground_truths)
        tuner.train(output_dir)
    """

    def __init__(self, config: Optional[T5PostProcessorConfig] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for T5 fine-tuning")

        self.config = config or T5PostProcessorConfig()
        self.device = torch.device(self.config.device)

        print(f"Initializing T5 fine-tuner with model: {self.config.model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)

        self.train_dataset = None
        self.eval_dataset = None

    def prepare_data(self,
                     train_sources: List[str],
                     train_targets: List[str],
                     eval_sources: Optional[List[str]] = None,
                     eval_targets: Optional[List[str]] = None,
                     layout_tokens: Optional[str] = None) -> None:
        """
        Prepare datasets for training.

        Args:
            train_sources: OCR predictions for training
            train_targets: Ground truth for training
            eval_sources: OCR predictions for evaluation (optional)
            eval_targets: Ground truth for evaluation (optional)
            layout_tokens: Layout tokens to remove before training
        """
        # Remove layout tokens for cleaner training
        if layout_tokens:
            pattern = re.compile(f'[{re.escape(layout_tokens)}]')
            train_sources = [pattern.sub('', s) for s in train_sources]
            train_targets = [pattern.sub('', t) for t in train_targets]
            if eval_sources:
                eval_sources = [pattern.sub('', s) for s in eval_sources]
            if eval_targets:
                eval_targets = [pattern.sub('', t) for t in eval_targets]

        # Create datasets
        self.train_dataset = T5CorrectionDataset(
            train_sources, train_targets,
            self.tokenizer,
            self.config.max_source_length,
            self.config.max_target_length
        )

        if eval_sources and eval_targets:
            self.eval_dataset = T5CorrectionDataset(
                eval_sources, eval_targets,
                self.tokenizer,
                self.config.max_source_length,
                self.config.max_target_length
            )

        print(f"Prepared {len(self.train_dataset)} training samples")
        if self.eval_dataset:
            print(f"Prepared {len(self.eval_dataset)} evaluation samples")

    def train(self,
              output_dir: str,
              num_epochs: int = 3,
              learning_rate: float = 5e-5,
              warmup_steps: int = 500,
              save_steps: int = 1000,
              eval_steps: int = 500,
              logging_steps: int = 100,
              fp16: bool = True) -> None:
        """
        Fine-tune the T5 model.

        Args:
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            fp16: Use mixed precision training
        """
        if self.train_dataset is None:
            raise ValueError("No training data. Call prepare_data() first.")

        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if self.eval_dataset else None,
            eval_strategy="steps" if self.eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if self.eval_dataset else False,
            fp16=fp16 and torch.cuda.is_available(),
            dataloader_num_workers=4,
            report_to=["tensorboard"],
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        print(f"Starting T5 fine-tuning...")
        print(f"  Output: {output_dir}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")

        trainer.train()

        # Save final model
        final_path = os.path.join(output_dir, "final")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Model saved to {final_path}")


class T5CorrectionDataset(torch.utils.data.Dataset):
    """Dataset for T5 OCR correction fine-tuning"""

    def __init__(self, sources: List[str], targets: List[str],
                 tokenizer, max_source_length: int, max_target_length: int):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = f"correct: {self.sources[idx]}"
        target = self.targets[idx]

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return {
            "input_ids": source_encoding.input_ids.squeeze(),
            "attention_mask": source_encoding.attention_mask.squeeze(),
            "labels": labels
        }


def create_training_data_from_predictions(
    prediction_file: str,
    ground_truth_file: str,
    output_dir: str,
    layout_tokens: Optional[str] = None,
    min_cer_threshold: float = 0.01,
    max_cer_threshold: float = 0.5
) -> Tuple[List[str], List[str]]:
    """
    Create training data for T5 from HAND prediction outputs.

    Args:
        prediction_file: Path to file with OCR predictions
        ground_truth_file: Path to file with ground truths
        output_dir: Directory to save processed data
        layout_tokens: Layout tokens to remove
        min_cer_threshold: Minimum CER to include (skip perfect matches)
        max_cer_threshold: Maximum CER to include (skip very bad predictions)

    Returns:
        Tuple of (sources, targets) lists
    """
    import pickle
    import editdistance

    # Load data
    with open(prediction_file, 'rb') as f:
        predictions = pickle.load(f)

    with open(ground_truth_file, 'rb') as f:
        ground_truths = pickle.load(f)

    sources = []
    targets = []

    # Remove layout tokens pattern
    pattern = None
    if layout_tokens:
        pattern = re.compile(f'[{re.escape(layout_tokens)}]')

    for pred, gt in zip(predictions, ground_truths):
        # Clean layout tokens
        if pattern:
            pred_clean = pattern.sub('', pred)
            gt_clean = pattern.sub('', gt)
        else:
            pred_clean = pred
            gt_clean = gt

        # Skip empty
        if not gt_clean.strip():
            continue

        # Calculate CER
        cer = editdistance.eval(pred_clean, gt_clean) / max(len(gt_clean), 1)

        # Filter by CER threshold
        if min_cer_threshold <= cer <= max_cer_threshold:
            sources.append(pred_clean)
            targets.append(gt_clean)

    print(f"Created {len(sources)} training pairs from predictions")
    print(f"  Filtered by CER: [{min_cer_threshold:.2%}, {max_cer_threshold:.2%}]")

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 't5_training_data.pkl'), 'wb') as f:
        pickle.dump({'sources': sources, 'targets': targets}, f)

    return sources, targets


# Convenience function for integration with HAND pipeline
def load_t5_processor(model_path: str,
                      layout_tokens: str = None,
                      device: str = None) -> T5PostProcessor:
    """
    Load a T5 post-processor for use in inference.

    Args:
        model_path: Path to fine-tuned T5 model
        layout_tokens: Layout tokens to preserve
        device: Device to use (cuda/cpu)

    Returns:
        Initialized T5PostProcessor
    """
    config = T5PostProcessorConfig(
        layout_tokens=layout_tokens,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return T5PostProcessor(config, model_path)
