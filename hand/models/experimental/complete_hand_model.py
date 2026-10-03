"""
Complete HAND Model
Integrates all components: Encoder, Decoder, MSAP Framework
Based on the complete HAND paper architecture
"""

import torch
import torch.nn as nn
from hand.models.experimental.hand_encoder import HAND_Encoder, ComplexityAssessmentNetwork
from hand.models.experimental.hand_decoder import HAND_Decoder
from hand.models.experimental.msap import (
    MSAPFramework,
    CurriculumLearningScheduler,
    ComplexityAwareLoss
)


class CompleteHAND(nn.Module):
    """
    Experimental full-model assembly (complexity network, adaptive queries, memory/sparse
    attention). Not the evaluated system and not used by any reported model.
    """
    def __init__(self, params):
        super(CompleteHAND, self).__init__()

        self.params = params
        self.enc_dim = params["enc_dim"]
        self.vocab_size = params["vocab_size"]
        self.charset = params.get("charset", [])
        self.tokens = params.get("tokens", {})

        # HAND Encoder (Section III-A)
        self.encoder = HAND_Encoder(params)

        # HAND Decoder (Section III-B)
        self.decoder = HAND_Decoder(params)

        # MSAP Framework (Section IV-C)
        self.msap = MSAPFramework(params)

        # 2D Positional Encoding for encoder output
        from hand.models.baseline.attention import PositionalEncoding2D
        self.pe_2d = PositionalEncoding2D(
            self.enc_dim,
            params.get("pe_h_max", 500),
            params.get("pe_w_max", 1000),
            params["device"]
        )

        # Curriculum Learning Scheduler
        self.curriculum_scheduler = CurriculumLearningScheduler(params)

        # Complexity-Aware Loss
        self.criterion = ComplexityAwareLoss(params)

        self.current_epoch = 0

    def forward(self, images, tokens, token_len=None, line_indices=None,
                index_in_lines=None, epoch=None):
        """
        Forward pass through complete HAND model

        Args:
            images: input document images [B, C, H, W]
            tokens: target tokens [B, T]
            token_len: length of each token sequence
            line_indices: line position indices
            index_in_lines: position within lines
            epoch: current training epoch

        Returns:
            logits: output predictions
            complexity: complexity scores
            attention_weights: attention weights
            losses: dictionary of losses
        """
        device = images.device
        batch_size = images.size(0)

        if epoch is None:
            epoch = self.current_epoch

        # Algorithm 2: First Pass - Feature Extraction
        # Equation 1-5: Pass through HAND encoder
        f_base = self.encoder(images)  # [B, C, H, W]

        # Add 2D positional encoding (Equation 6)
        f_2d = self.pe_2d(f_base)

        # First pass features with warmup (Equation 25-26)
        f1 = self.msap.first_pass_features(f_2d, epoch)

        # Assess document complexity (Equation 23)
        complexity = self.msap.assess_complexity(f1)

        # Algorithm 3: Second Pass - Complexity-aware processing
        f2, alpha_cl, beta_cl, omega_cl = self.msap.second_pass_features(f1, complexity)

        # Create masks for decoder
        tgt_mask = self._generate_square_subsequent_mask(tokens.size(1), device)

        # Prepare key padding masks
        if token_len is not None:
            tgt_key_padding_mask = self._generate_padding_mask(tokens, token_len)
        else:
            tgt_key_padding_mask = None

        # Generate memory padding mask for encoder features
        memory_key_padding_mask = None  # Can be added if needed

        # Pass through HAND decoder
        logits, attention_weights, decoder_output = self.decoder(
            tokens=tokens,
            features_2d=f2,
            token_len=token_len,
            features_size=f2.shape,
            line_indices=line_indices,
            index_in_lines=index_in_lines,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return {
            'logits': logits,
            'complexity': complexity,
            'attention_weights': attention_weights,
            'alpha_cl': alpha_cl,
            'beta_cl': beta_cl,
            'omega_cl': omega_cl,
            'features': f2
        }

    def compute_loss(self, outputs, targets, layout_targets=None):
        """
        Compute HAND loss using complexity-aware loss function

        Args:
            outputs: model outputs dictionary
            targets: target text tokens
            layout_targets: target layout labels (optional)

        Returns:
            total_loss: combined loss
            loss_dict: dictionary of individual losses
        """
        logits = outputs['logits']
        complexity = outputs['complexity']
        features = outputs['features']

        # Compute loss (Equations 32-36)
        total_loss, loss_dict = self.criterion(
            text_logits=logits,
            text_targets=targets,
            layout_logits=None,  # Can be added if layout prediction is included
            layout_targets=layout_targets,
            complexity=complexity,
            features=features
        )

        # Add complexity score to loss dict for monitoring
        loss_dict['complexity_score'] = complexity.mean().item()
        loss_dict['alpha_cl'] = outputs['alpha_cl'].mean().item()
        loss_dict['beta_cl'] = outputs['beta_cl'].mean().item()

        return total_loss, loss_dict

    def _generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _generate_padding_mask(self, tokens, token_len):
        """Generate padding mask for tokens"""
        batch_size, max_len = tokens.size()
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=tokens.device)

        for i, length in enumerate(token_len):
            if length < max_len:
                mask[i, length:] = True

        return mask

    def update_epoch(self, epoch):
        """Update current epoch for curriculum learning and warmup"""
        self.current_epoch = epoch

        # Check if should switch curriculum level
        switched, current_level = self.curriculum_scheduler.should_switch_level(epoch)

        if switched:
            print(f"Switching to curriculum level: {current_level}")

        return current_level

    def get_curriculum_batch_size(self):
        """Get adaptive batch size based on current curriculum level"""
        level_idx, level = self.curriculum_scheduler.get_current_level(self.current_epoch)
        return self.curriculum_scheduler.get_batch_size(level_idx)

    @torch.no_grad()
    def generate(self, images, max_len=500, temperature=1.0):
        """
        Generate text from images (inference mode)

        Args:
            images: input images [B, C, H, W]
            max_len: maximum sequence length
            temperature: sampling temperature

        Returns:
            generated_strings: list of predicted text strings
        """
        self.eval()
        device = images.device
        batch_size = images.size(0)

        # Encode images
        f_base = self.encoder(images)
        f_2d = self.pe_2d(f_base)
        f1 = self.msap.first_pass_features(f_2d, self.current_epoch)
        complexity = self.msap.assess_complexity(f1)
        f2, _, _, _ = self.msap.second_pass_features(f1, complexity)

        # Get start and end tokens
        sos_token = self.tokens.get('start', self.vocab_size + 1)
        eos_token = self.tokens.get('end', self.vocab_size)

        # Start with <SOS> token
        generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Create mask
            tgt_mask = self._generate_square_subsequent_mask(generated.size(1), device)

            # Forward through decoder
            logits, _, _ = self.decoder(
                tokens=generated,
                features_2d=f2,
                tgt_mask=tgt_mask
            )

            # Get last token logits
            next_token_logits = logits[:, :, -1] / temperature

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for <EOS> token
            if (next_token == eos_token).all():
                break

        # Convert tokens to strings
        generated_strings = self._decode_tokens(generated)

        return generated_strings

    def _decode_tokens(self, token_sequences):
        """
        Convert token sequences to strings

        Args:
            token_sequences: tensor of shape [B, T] containing token indices

        Returns:
            list of decoded strings
        """
        batch_size = token_sequences.size(0)
        decoded_strings = []

        # Get special tokens
        sos_token = self.tokens.get('start', self.vocab_size + 1)
        eos_token = self.tokens.get('end', self.vocab_size)
        pad_token = self.tokens.get('pad', self.vocab_size + 2)

        for i in range(batch_size):
            tokens = token_sequences[i].cpu().tolist()

            # Remove special tokens and padding
            decoded_chars = []
            for token_id in tokens:
                # Skip special tokens
                if token_id in [sos_token, eos_token, pad_token]:
                    if token_id == eos_token:
                        break  # Stop at EOS
                    continue

                # Convert valid tokens to characters
                if 0 <= token_id < len(self.charset):
                    decoded_chars.append(self.charset[token_id])

            decoded_strings.append(''.join(decoded_chars))

        return decoded_strings


def create_hand_model(params):
    """
    Factory function to create HAND model with proper initialization

    Args:
        params: model parameters dictionary

    Returns:
        model: initialized HAND model
    """
    model = CompleteHAND(params)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    return model
