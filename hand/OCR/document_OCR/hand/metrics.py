"""
HTR Evaluation Metrics
Character Error Rate (CER) and Word Error Rate (WER)
"""

import editdistance
import torch
import numpy as np


class HTRMetrics:
    """
    Handwritten Text Recognition Metrics
    Implements CER and WER calculations
    """

    def __init__(self, charset, pad_token=-1, eos_token=None, sos_token=None):
        """
        Args:
            charset: list of characters
            pad_token: padding token index
            eos_token: end-of-sequence token index
            sos_token: start-of-sequence token index
        """
        self.charset = charset
        self.pad_token = pad_token
        self.eos_token = eos_token if eos_token is not None else len(charset)
        self.sos_token = sos_token if sos_token is not None else len(charset) + 1

        # Create index to character mapping
        self.idx2char = {i: c for i, c in enumerate(charset)}

    def decode_tokens(self, tokens, debug=False):
        """
        Decode token indices to string

        Args:
            tokens: tensor of shape [B, T] or list of token indices
            debug: if True, print debug info for empty strings

        Returns:
            list of decoded strings
        """
        if isinstance(tokens, torch.Tensor):
            # Convert to list without numpy (compatible with multiprocessing)
            if len(tokens.shape) == 1:
                tokens = [tokens.cpu().tolist()]
            else:
                tokens = tokens.cpu().tolist()
        elif len(tokens.shape) == 1:
            # Single sequence
            tokens = [tokens]

        decoded_strings = []
        for seq_idx, seq in enumerate(tokens):
            chars = []
            for token_idx, idx in enumerate(seq):
                idx = int(idx)
                # Stop at EOS, skip padding and SOS
                if idx == self.eos_token:
                    break
                if idx == self.pad_token or idx == self.sos_token:
                    continue
                # Get character
                if idx in self.idx2char:
                    chars.append(self.idx2char[idx])
                else:
                    # Unknown token - use placeholder
                    chars.append('�')

            result = ''.join(chars)

            # Debug: warn about empty strings
            if debug and len(result) == 0 and seq_idx < 2:  # Only for first 2 samples
                non_pad_tokens = [int(t) for t in seq if int(t) != self.pad_token]
                print(f"[METRICS DEBUG] Empty decoded string for sequence {seq_idx}!")
                print(f"  Non-padding tokens: {non_pad_tokens[:20]}")
                print(f"  EOS token: {self.eos_token}, SOS token: {self.sos_token}")
                if len(non_pad_tokens) > 0:
                    print(f"  First token: {non_pad_tokens[0]} (is_SOS: {non_pad_tokens[0] == self.sos_token}, is_EOS: {non_pad_tokens[0] == self.eos_token})")
                    if len(non_pad_tokens) > 1:
                        print(f"  Second token: {non_pad_tokens[1]} (in charset: {non_pad_tokens[1] in self.idx2char})")

            decoded_strings.append(result)

        return decoded_strings

    def decode_predictions(self, logits):
        """
        Decode logits to strings using greedy decoding

        Args:
            logits: tensor of shape [B, num_classes, T]

        Returns:
            list of decoded strings
        """
        # Get predictions via argmax
        predictions = torch.argmax(logits, dim=1)  # [B, T]
        return self.decode_tokens(predictions)

    def compute_cer(self, predictions, targets):
        """
        Compute Character Error Rate

        CER = (Substitutions + Deletions + Insertions) / Total Characters

        Args:
            predictions: list of predicted strings
            targets: list of target strings

        Returns:
            float: CER value
        """
        if isinstance(predictions, torch.Tensor):
            predictions = self.decode_predictions(predictions)
        elif isinstance(predictions[0], int):
            predictions = self.decode_tokens(predictions)

        if isinstance(targets, torch.Tensor):
            targets = self.decode_tokens(targets)
        elif isinstance(targets[0], int):
            targets = self.decode_tokens(targets)

        total_distance = 0
        total_length = 0

        for pred, targ in zip(predictions, targets):
            # Compute edit distance
            distance = editdistance.eval(pred, targ)
            total_distance += distance
            total_length += len(targ)

        if total_length == 0:
            return 0.0

        cer = total_distance / total_length
        return cer

    def compute_wer(self, predictions, targets):
        """
        Compute Word Error Rate

        WER = (Substitutions + Deletions + Insertions) / Total Words

        Args:
            predictions: list of predicted strings
            targets: list of target strings

        Returns:
            float: WER value
        """
        if isinstance(predictions, torch.Tensor):
            predictions = self.decode_predictions(predictions)
        elif isinstance(predictions[0], int):
            predictions = self.decode_tokens(predictions)

        if isinstance(targets, torch.Tensor):
            targets = self.decode_tokens(targets)
        elif isinstance(targets[0], int):
            targets = self.decode_tokens(targets)

        total_distance = 0
        total_words = 0

        for pred, targ in zip(predictions, targets):
            # Split into words
            pred_words = pred.split()
            targ_words = targ.split()

            # Compute edit distance at word level
            distance = editdistance.eval(pred_words, targ_words)
            total_distance += distance
            total_words += len(targ_words)

        if total_words == 0:
            return 0.0

        wer = total_distance / total_words
        return wer

    def compute_metrics(self, logits, targets, debug=False):
        """
        Compute both CER and WER

        Args:
            logits: tensor [B, num_classes, T] - model predictions
            targets: tensor [B, T] - ground truth tokens
            debug: if True, print debug info

        Returns:
            dict with 'cer' and 'wer' keys
        """
        predictions = self.decode_predictions(logits)
        target_strings = self.decode_tokens(targets, debug=debug)

        cer = self.compute_cer(predictions, target_strings)
        wer = self.compute_wer(predictions, target_strings)

        return {
            'cer': cer * 100,  # Convert to percentage
            'wer': wer * 100,
            'predictions': predictions[:3],  # Sample predictions for logging
            'targets': target_strings[:3]    # Sample targets for logging
        }


# Standalone functions for backward compatibility
def compute_cer(predictions, targets, charset):
    """Standalone CER computation"""
    metrics = HTRMetrics(charset)
    return metrics.compute_cer(predictions, targets)


def compute_wer(predictions, targets, charset):
    """Standalone WER computation"""
    metrics = HTRMetrics(charset)
    return metrics.compute_wer(predictions, targets)
