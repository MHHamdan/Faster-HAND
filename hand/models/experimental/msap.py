"""
Multi-Scale Adaptive Processing (MSAP) Framework
Based on HAND paper Section IV-C and Algorithms 1-3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hand.models.experimental.hand_encoder import ComplexityAssessmentNetwork


class MSAPFramework(nn.Module):
    """
    Multi-Scale Adaptive Processing Framework
    Implements Algorithm 1 from HAND paper
    """
    def __init__(self, params):
        super(MSAPFramework, self).__init__()

        self.params = params
        self.enc_dim = params["enc_dim"]

        # Complexity Assessment Network (Equation 23)
        self.complexity_network = ComplexityAssessmentNetwork(
            enc_dim=self.enc_dim,
            hidden_dim=128
        )

        # Parameters for complexity-dependent scaling (Equation 28)
        self.register_buffer('alpha_0', torch.tensor(0.1))
        self.register_buffer('beta_0', torch.tensor(0.1))
        self.register_buffer('omega_0', torch.tensor(1.0))

        self.register_buffer('gamma_alpha', torch.tensor(0.5))
        self.register_buffer('gamma_beta', torch.tensor(0.5))
        self.register_buffer('gamma_omega', torch.tensor(0.3))

        self.register_buffer('delta_alpha', torch.tensor(10.0))
        self.register_buffer('delta_beta', torch.tensor(10.0))
        self.register_buffer('delta_omega', torch.tensor(10.0))

        self.register_buffer('theta_alpha', torch.tensor(0.5))
        self.register_buffer('theta_beta', torch.tensor(0.5))
        self.register_buffer('theta_omega', torch.tensor(0.5))

        # Warmup parameters (Equation 26)
        self.warmup_epochs = params.get('warmup_epochs', 150)

    def assess_complexity(self, features):
        """
        Equation 23: C(x) = φ(Encoder(x)) ∈ [0, 1]

        Args:
            features: encoder output [B, C, H, W]
        Returns:
            complexity: complexity scores [B, 1]
        """
        return self.complexity_network(features)

    def compute_scaling_factors(self, complexity):
        """
        Equation 28: Complexity-dependent scaling
        α(C_l), β(C_l), ω(C_l)

        Args:
            complexity: complexity scores [B, 1]
        Returns:
            alpha, beta, omega: scaling factors
        """
        # Alpha scaling
        alpha = self.alpha_0 * (
            (1 + self.gamma_alpha * complexity) /
            (1 + torch.exp(self.delta_alpha * (complexity - self.theta_alpha)))
        )

        # Beta scaling
        beta = self.beta_0 * (
            (1 + self.gamma_beta * complexity) /
            (1 + torch.exp(self.delta_beta * (complexity - self.theta_beta)))
        )

        # Omega scaling
        omega = self.omega_0 * (
            (1 + self.gamma_omega * complexity) /
            (1 + torch.exp(self.delta_omega * (complexity - self.theta_omega)))
        )

        return alpha, beta, omega

    def first_pass_features(self, features, epoch):
        """
        Algorithm 2: First Pass Feature Extraction (Equation 25-26)

        Args:
            features: base features from encoder [B, C, H, W]
            epoch: current epoch number
        Returns:
            f1: position-aware features
        """
        # Equation 26: Warmup schedule for positional encoding
        alpha_e = self.alpha_0 * (1 + self.gamma_alpha * min(1.0, epoch / self.warmup_epochs))

        # Apply 2D positional encoding (already done in encoder)
        # Here we modulate the influence
        f1 = features * (1 + alpha_e.item())

        return f1

    def second_pass_features(self, f1, complexity):
        """
        Algorithm 3: Second Pass Feature Processing (Equation 27-28)

        Args:
            f1: first pass features
            complexity: document complexity score
        Returns:
            f2: refined features with complexity-aware scaling
        """
        # Compute scaling factors (Equation 28)
        alpha_cl, beta_cl, omega_cl = self.compute_scaling_factors(complexity)

        # Scale features
        f2 = f1 * alpha_cl.view(-1, 1, 1, 1)

        return f2, alpha_cl, beta_cl, omega_cl

    def adaptive_query_generation(self, token_embeddings, doc_context, rel_pos_enc,
                                  alpha_cl, beta_cl):
        """
        Equation 27: Adaptive Query Generation
        q_M^i = E(y_M^i) + α_Cl * P_doc^M + β_Cl * R_M^i

        Args:
            token_embeddings: embedded tokens E(y_M^i)
            doc_context: document-level context P_doc^M
            rel_pos_enc: relative positional encoding R_M^i
            alpha_cl, beta_cl: complexity-dependent scaling factors
        Returns:
            queries: adaptive queries
        """
        # Base token embeddings
        queries = token_embeddings

        # Add scaled document context
        if doc_context is not None:
            queries = queries + alpha_cl.view(-1, 1, 1) * doc_context

        # Add scaled relative positional encoding
        if rel_pos_enc is not None:
            queries = queries + beta_cl.view(-1, 1, 1) * rel_pos_enc

        return queries

    def compute_adaptive_batch_size(self, base_batch_size, level, gamma=0.8, min_batch=1):
        """
        Equation 21: Adaptive batch sizing
        B_l = max(⌊B_0 · γ^l⌋, B_min)

        Args:
            base_batch_size: initial batch size B_0
            level: curriculum level l
            gamma: decay factor
            min_batch: minimum batch size
        Returns:
            batch_size: adapted batch size
        """
        batch_size = max(int(base_batch_size * (gamma ** level)), min_batch)
        return batch_size


class CurriculumLearningScheduler:
    """
    Hierarchical Curriculum Learning Scheduler
    Implements hierarchical curriculum from HAND paper
    Note: Using 4-level curriculum (line→page→double_page→triple_page)
    as paragraph-level formatted dataset is not available
    """
    def __init__(self, params):
        # Using 4 levels instead of original 5 (paragraph level unavailable)
        self.levels = ['line', 'page', 'double_page', 'triple_page']
        self.current_level = 0

        # Curriculum parameters (Equation 19)
        # Adjusted epoch distribution for 4 levels
        self.level_epochs = params.get('level_epochs', {
            'line': 150,          # Extended from 100
            'page': 200,          # Keep same
            'double_page': 250,   # Keep same
            'triple_page': 300    # Keep same
        })

        # Adaptive batch sizing parameters
        self.base_batch_size = params.get('batch_size', 8)
        self.gamma = params.get('batch_decay', 0.8)

        # Level-specific loss weights (Equation 19)
        # Adjusted for 4 levels
        self.alpha_weights = {
            'line': 1.0,
            'page': 1.3,          # Adjusted from 1.5
            'double_page': 1.7,   # Adjusted from 1.8
            'triple_page': 2.0
        }

    def get_current_level(self, epoch):
        """Get current curriculum level based on epoch"""
        cumulative_epochs = 0
        for i, level in enumerate(self.levels):
            cumulative_epochs += self.level_epochs[level]
            if epoch < cumulative_epochs:
                return i, level
        return len(self.levels) - 1, self.levels[-1]

    def get_batch_size(self, level_idx):
        """Get adaptive batch size for current level"""
        return max(int(self.base_batch_size * (self.gamma ** level_idx)), 1)

    def get_loss_weight(self, level):
        """Get loss weight for current level (Equation 19)"""
        return self.alpha_weights.get(level, 1.0)

    def should_switch_level(self, epoch):
        """Check if should switch to next curriculum level"""
        cumulative_epochs = 0
        for level in self.levels[:self.current_level + 1]:
            cumulative_epochs += self.level_epochs[level]

        if epoch >= cumulative_epochs and self.current_level < len(self.levels) - 1:
            self.current_level += 1
            return True, self.levels[self.current_level]
        return False, self.levels[self.current_level]


class ComplexityAwareLoss(nn.Module):
    """
    Complexity-Aware Loss Function
    Implements Equations 32-36 from HAND paper
    """
    def __init__(self, params):
        super(ComplexityAwareLoss, self).__init__()

        # Loss weights (Equation 32)
        # IMPROVED: Focus on text since we don't have layout labels
        self.lambda_layout_0 = params.get('lambda_layout', 0.0)  # Disabled (no layout labels)
        self.lambda_text_0 = params.get('lambda_text', 1.0)      # Full weight on text
        self.lambda_c_0 = params.get('lambda_complexity', 0.1)   # Complexity guidance

        # Label smoothing for better generalization (Research: 2024 Best Practices)
        self.label_smoothing = params.get('label_smoothing', 0.1)

        # Dynamic weighting parameters (Equation 36)
        self.gamma_layout = 0.3
        self.gamma_text = 0.2
        self.delta_layout = 10.0
        self.delta_text = 10.0
        self.theta_layout = 0.5
        self.theta_text = 0.5

        # Gradient regularization weight
        self.lambda_reg = 0.01

    def compute_dynamic_weights(self, complexity):
        """
        Equation 36: Dynamic weighting based on complexity
        λ_k(C_l) = λ_k^0 · (1 + γ_k C_l) / (1 + exp(δ_k(C_l - θ_k)))
        """
        # Layout weight
        lambda_layout = self.lambda_layout_0 * (
            (1 + self.gamma_layout * complexity) /
            (1 + torch.exp(self.delta_layout * (complexity - self.theta_layout)))
        )

        # Text weight
        lambda_text = self.lambda_text_0 * (
            (1 + self.gamma_text * complexity) /
            (1 + torch.exp(self.delta_text * (complexity - self.theta_text)))
        )

        return lambda_layout, lambda_text

    def forward(self, text_logits, text_targets, layout_logits, layout_targets,
                complexity, features):
        """
        Compute total loss (Equation 32)

        Args:
            text_logits: predicted text
            text_targets: ground truth text
            layout_logits: predicted layout
            layout_targets: ground truth layout
            complexity: complexity scores
            features: encoder features (for complexity loss)
        Returns:
            total_loss, loss_dict
        """
        # Get dynamic weights
        lambda_layout, lambda_text = self.compute_dynamic_weights(complexity.mean())

        # Equation 34: Text recognition loss (cross-entropy)
        if text_logits is not None and text_targets is not None:
            # text_logits: [B, vocab_size, T]
            # text_targets: [B, T]
            # Need to permute logits to [B, T, vocab_size] for cross_entropy
            text_logits_permuted = text_logits.permute(0, 2, 1)  # [B, T, vocab_size]
            num_classes = text_logits_permuted.size(-1)

            # Validate target range (excluding padding -1)
            valid_targets = text_targets[text_targets != -1]
            if len(valid_targets) > 0:
                max_target = valid_targets.max().item()
                min_target = valid_targets.min().item()
                if max_target >= num_classes or min_target < 0:
                    raise ValueError(
                        f"Target out of range! Targets: [{min_target}, {max_target}], "
                        f"but num_classes={num_classes}. Valid range is [0, {num_classes-1}] or -1 (padding)."
                    )

            text_loss = F.cross_entropy(
                text_logits_permuted.reshape(-1, text_logits_permuted.size(-1)),
                text_targets.reshape(-1),
                ignore_index=-1,
                label_smoothing=self.label_smoothing  # IMPROVED: Add label smoothing
            )
        else:
            text_loss = torch.tensor(0.0, device=features.device)

        # Equation 33: Layout loss (if applicable)
        if layout_logits is not None and layout_targets is not None:
            layout_loss = F.cross_entropy(
                layout_logits.reshape(-1, layout_logits.size(-1)),
                layout_targets.reshape(-1),
                ignore_index=-1
            )
        else:
            layout_loss = torch.tensor(0.0, device=features.device)

        # Equation 35: Complexity loss with gradient regularization
        if complexity is not None:
            # MSE term for complexity prediction
            target_complexity = self._estimate_target_complexity(features)
            # Ensure shapes match for MSE loss
            complexity_flat = complexity.view(-1)
            target_flat = target_complexity.view(-1)
            complexity_mse = F.mse_loss(complexity_flat, target_flat)

            # Gradient penalty term - skip if features doesn't require grad
            # (features may be detached in some cases)
            if features.requires_grad:
                try:
                    grad_outputs = torch.ones_like(complexity)
                    gradients = torch.autograd.grad(
                        outputs=complexity,
                        inputs=features,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                        allow_unused=True
                    )[0]
                    if gradients is not None:
                        gradient_penalty = torch.norm(gradients, p=1, dim=(1, 2, 3)).mean()
                    else:
                        gradient_penalty = torch.tensor(0.0, device=features.device)
                except RuntimeError:
                    # If gradient computation fails, skip it
                    gradient_penalty = torch.tensor(0.0, device=features.device)
            else:
                gradient_penalty = torch.tensor(0.0, device=features.device)

            complexity_loss = complexity_mse + self.lambda_reg * gradient_penalty
        else:
            complexity_loss = torch.tensor(0.0, device=features.device)

        # Equation 32: Total loss
        total_loss = (lambda_layout * layout_loss +
                     lambda_text * text_loss +
                     self.lambda_c_0 * complexity_loss)

        loss_dict = {
            'total': total_loss.item(),
            'text': text_loss.item(),
            'layout': layout_loss.item(),
            'complexity': complexity_loss.item(),
            'lambda_layout': lambda_layout.item() if torch.is_tensor(lambda_layout) else lambda_layout,
            'lambda_text': lambda_text.item() if torch.is_tensor(lambda_text) else lambda_text
        }

        return total_loss, loss_dict

    def _estimate_target_complexity(self, features):
        """Estimate target complexity based on feature statistics"""
        # Simple heuristic: use feature variance as complexity indicator
        var = torch.var(features, dim=(2, 3), keepdim=True)
        complexity = torch.sigmoid(var.mean(dim=1, keepdim=True))
        return complexity.squeeze()
