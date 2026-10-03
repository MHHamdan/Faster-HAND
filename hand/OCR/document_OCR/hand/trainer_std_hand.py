from hand.OCR.document_OCR.hand.trainer_hand import Manager as DANManager
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from hand.OCR.ocr_utils import LM_ind_to_str
from torch.cuda.amp import autocast
import os
import time

# Debug NaN/Inf probes, opt-in (HAND_DEBUG_CHECKS=1): each is a host-device synchronisation.
DEBUG_CHECKS = os.environ.get("HAND_DEBUG_CHECKS", "0") == "1"


class Manager(DANManager):

    def __init__(self, params):
        super(DANManager, self).__init__(params)

    def get_line_indices_from_tokens(self, batch_tokens):
        """
        Line index and index-within-line for every token position (Faster-DAN two-step
        positional encoding), computed on the device without any host synchronisation.

        Semantics, identical to the reference loop below: a "character" token is one that
        is not the newline and whose id is <= len(char_only_set) -- or, with
        layout_token_identity set, one that is neither a semantic token nor a special
        token, which is the same set minus the positional off-by-one. Padding tokens change no
        state and repeat the current values. A character token that follows a
        non-character token opens a new line (line index +1, index-in-line reset to 0);
        every other non-padding token advances index-in-line by one. Before the first line
        the line index is -1.
        """
        tokens = batch_tokens
        pad = tokens == self.dataset.tokens["pad"]
        nonpad = ~pad
        new_line_token = self.dataset.charset.index("\n")
        if getattr(self.dataset, "layout_token_identity", False):
            layout = torch.tensor(sorted(self.dataset.layout_token_ids),
                                  device=tokens.device, dtype=tokens.dtype)
            is_char = ((tokens != new_line_token) & (tokens < len(self.dataset.charset))
                       & ~torch.isin(tokens, layout) & nonpad)
        else:
            is_char = (tokens != new_line_token) & (tokens <= len(self.dataset.char_only_set)) & nonpad

        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        # position of the previous non-padding token (state carrier), -1 if none
        last_np = torch.cummax(torch.where(nonpad, pos, torch.full_like(pos, -1)), dim=1).values
        prev_np = torch.cat([torch.full((b, 1), -1, device=tokens.device, dtype=last_np.dtype), last_np[:, :-1]], dim=1)
        prev_is_char = torch.where(prev_np >= 0, torch.gather(is_char, 1, prev_np.clamp(min=0)), torch.ones_like(is_char))
        new_line = is_char & ~prev_is_char                        # only at non-pad char tokens
        line_indices = torch.cumsum(new_line.to(tokens.dtype), dim=1) - 1

        cnt = torch.cumsum(nonpad.to(tokens.dtype), dim=1)      # non-pad tokens seen so far
        seg = torch.cumsum(new_line.to(tokens.dtype), dim=1)    # segment id, 0 before first line
        # count at the start of each segment; segment 0 starts at count 0
        start_cnt = torch.zeros((b, t + 1), device=tokens.device, dtype=tokens.dtype)
        start_cnt.scatter_(1, torch.where(new_line, seg, torch.zeros_like(seg)), torch.where(new_line, cnt, torch.zeros_like(cnt)))
        start_cnt[:, 0] = 0
        index_in_lines = cnt - torch.gather(start_cnt, 1, seg)
        return line_indices, index_in_lines

    def get_line_indices_from_tokens_reference(self, batch_tokens):
        """Original per-token Python loop (one host sync per token). Kept for the
        equivalence test only."""
        batch_line_indices = list()
        batch_index_in_lines = list()
        identity = getattr(self.dataset, "layout_token_identity", False)
        layout_ids = getattr(self.dataset, "layout_token_ids", frozenset())
        for b in range(batch_tokens.size(0)):
            tokens = batch_tokens[b]
            new_line_token = self.dataset.charset.index("\n")
            line_indices = list()
            index_in_lines = list()
            last_token_char = True
            line_index = -1
            index_in_line = 0
            for t in tokens:
                if t == self.dataset.tokens["pad"]:
                    pass
                elif t != new_line_token and (
                        (int(t) not in layout_ids and t < len(self.dataset.charset))
                        if identity else t <= len(self.dataset.char_only_set)):
                    if not last_token_char:
                        line_index += 1
                        index_in_line = 0
                    else:
                        index_in_line += 1
                    last_token_char = True
                else:
                    last_token_char = False
                    index_in_line += 1

                index_in_lines.append(index_in_line)
                line_indices.append(line_index)
            batch_line_indices.append(torch.tensor(line_indices, dtype=batch_tokens.dtype, device=batch_tokens.device))
            batch_index_in_lines.append(torch.tensor(index_in_lines, dtype=batch_tokens.dtype, device=batch_tokens.device))
        return torch.stack(batch_line_indices, dim=0), torch.stack(batch_index_in_lines, dim=0)

    def train_batch(self, batch_data, metric_names):
        # Label smoothing (B-3a). Default 0.0 -- every existing run and every reported number is
        # produced by the unsmoothed loss. It is a flag because the model is measurably
        # over-confident: X2 records teacher-forced top-1 probability 0.9927 against accuracy
        # 0.9572 on the full-budget checkpoint (a 3.55 pp calibration gap) with an output entropy
        # of 0.0195 nats over 99 symbols, and the gap is WIDER than the 500 k checkpoint's
        # 2.28 pp -- longer training made it worse, which is the setting this regulariser targets.
        fn_loss_ce = CrossEntropyLoss(
            ignore_index=self.dataset.tokens["pad"], reduction="mean",
            label_smoothing=self.params["training_params"].get("label_smoothing", 0.0))
        x = batch_data["imgs"].to(self.device, non_blocking=True)
        y = batch_data["labels"].to(self.device, non_blocking=True)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]
        str_y = batch_data["raw_labels"]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate)
        else:
            simulated_y_pred = y

        with autocast(enabled=self.params["training_params"]["use_amp"]):

            raw_features = self.train_encoder(x, batch_data)
            features_size = raw_features.size()
            b, c, h, w = features_size

            # Debug: Check encoder output
            if DEBUG_CHECKS and (torch.isnan(raw_features).any() or torch.isinf(raw_features).any()):
                print(f"  NaN/Inf in encoder output!", flush=True)

            pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            # Debug: Check pos_features
            if DEBUG_CHECKS and (torch.isnan(pos_features).any() or torch.isinf(pos_features).any()):
                print(f"  NaN/Inf in pos_features!", flush=True)

            simulated_y_pred = simulated_y_pred[:, :-1]
            y = y[:, 1:]

            # --- Scheduled sampling -------------------------------------------------
            # `add_error_in_gt` substitutes UNIFORMLY RANDOM tokens, so the decoder only
            # ever learns to recover from random corruption. Its real inference-time
            # mistakes are systematic -- visually confusable characters -- which is a very
            # different distribution, and that gap is what the measured train/test spread
            # (train CER ~1.6% vs test 4.55% at page level) reflects.
            #
            # True scheduled sampling conditions on the model's OWN predictions: run a
            # no-grad teacher-forced pass, then replace a fraction of the decoder input
            # with what the model actually predicted there, and compute the loss on that
            # mixed input. Costs one extra forward pass, no extra backward.
            ss_rate = self.params["training_params"].get("scheduled_sampling_rate", 0.0)
            if ss_rate and ss_rate > 0:
                ss_li = ss_ii = None
                if self.params["model_params"].get("use_line_indices", False):
                    ss_li, ss_ii = self.get_line_indices_from_tokens(simulated_y_pred)
                with torch.no_grad():
                    _, ss_pred, _, _ = self.models["decoder"](
                        pos_features, pos_features, simulated_y_pred, reduced_size,
                        y_len, features_size, start=0,
                        line_indices=ss_li, index_in_lines=ss_ii,
                        padding_value=self.dataset.tokens["pad"])
                    model_tokens = torch.argmax(ss_pred, dim=1)          # (B, T)
                # Shift by one: the token fed at position t is the prediction made at t-1.
                shifted = torch.roll(model_tokens, shifts=1, dims=1)
                shifted[:, 0] = simulated_y_pred[:, 0]                   # keep <sot>
                take_model = torch.rand_like(simulated_y_pred, dtype=torch.float) < ss_rate
                take_model &= simulated_y_pred != self.dataset.tokens["pad"]
                simulated_y_pred = torch.where(take_model, shifted, simulated_y_pred)

            line_indices = None
            index_in_lines = None
            if "use_line_indices" in self.params["model_params"] and self.params["model_params"]["use_line_indices"]:
                line_indices, index_in_lines = self.get_line_indices_from_tokens(simulated_y_pred)
                # Debug: Check line_indices
                if DEBUG_CHECKS and (torch.isnan(line_indices.float()).any() or torch.isinf(line_indices.float()).any()):
                    print(f"  NaN/Inf in line_indices!", flush=True)
                if DEBUG_CHECKS and (torch.isnan(index_in_lines.float()).any() or torch.isinf(index_in_lines.float()).any()):
                    print(f"  NaN/Inf in index_in_lines!", flush=True)
                # Debug: Check for very large indices
                if DEBUG_CHECKS and (line_indices.max() > 1000 or index_in_lines.max() > 1000):
                    print(f"  Large indices: line_indices max={line_indices.max().item()}, index_in_lines max={index_in_lines.max().item()}", flush=True)
            output, pred, cache, weights = self.models["decoder"](pos_features, pos_features,
                                                                           simulated_y_pred,
                                                                           reduced_size,
                                                                           y_len,
                                                                           features_size,
                                                                           start=0,
                                                                          line_indices=line_indices,
                                                                          index_in_lines=index_in_lines,
                                                                           keep_all_weights=True,
                                                                            padding_value=self.dataset.tokens["pad"])
            loss = fn_loss_ce(pred, y)

            # One host fetch of the loss value serves both the finiteness guard below and
            # the reported metric (it used to be fetched three times).
            loss_value = loss.item()
            if not np.isfinite(loss_value):
                print(f"WARNING: NaN/Inf loss detected - skipping batch!", flush=True)
                print(f"  pred shape: {pred.shape}", flush=True)
                if not torch.isnan(pred).all():
                    print(f"  pred min/max: {pred[~torch.isnan(pred)].min().item():.4f}/{pred[~torch.isnan(pred)].max().item():.4f}", flush=True)
                print(f"  y shape: {y.shape}, y min/max: {y.min().item()}/{y.max().item()}", flush=True)
                print(f"  vocab_size (num classes): {pred.shape[1]}", flush=True)
                print(f"  y max should be < {pred.shape[1]}", flush=True)
                if torch.isnan(pred).any():
                    nan_count = torch.isnan(pred).sum().item()
                    print(f"  NaN found in predictions: {nan_count}/{pred.numel()} ({100*nan_count/pred.numel():.2f}%)", flush=True)
                if torch.isinf(pred).any():
                    inf_count = torch.isinf(pred).sum().item()
                    print(f"  Inf found in predictions: {inf_count}/{pred.numel()} ({100*inf_count/pred.numel():.2f}%)", flush=True)
                # Check if labels are out of range
                if y.max() >= pred.shape[1]:
                    print(f"  ERROR: Label index {y.max().item()} >= num_classes {pred.shape[1]}", flush=True)
                # Skip this batch - don't backprop NaN
                self.zero_optimizers()
                return {
                    "nb_samples": b,
                    "str_y": str_y,
                    "str_x": ["" for _ in range(b)],  # Empty predictions for skipped batch
                    "loss": 0.0,  # Don't contribute to loss
                    "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
                }

            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        self.backward_loss(loss)
        self.step_optimizers()
        self.zero_optimizers()

        values = {
            "nb_samples": b,
            "str_y": str_y,
            "str_x": str_x,
            "loss": loss_value,
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
        }

        return values

    def evaluate_batch(self, batch_data, metric_names):
        """Dispatch to beam search when a beam width > 1 is configured."""
        beam = self.params["training_params"].get("beam_size", 1) or 1
        if beam > 1:
            return self.evaluate_batch_beam(batch_data, metric_names, beam)
        return self.evaluate_batch_greedy(batch_data, metric_names)

    def evaluate_batch_beam(self, batch_data, metric_names, beam_size):
        """
        Beam-search decoding.

        The greedy loop commits to argmax at every step, which is unrecoverable: one wrong
        character early can derail an entire line. Beam search keeps `beam_size` hypotheses
        alive and scores them by total log-probability, normalised by length so that short
        hypotheses are not unfairly favoured.

        Layout: everything is flattened to (B*K, ...) so a single decoder call advances all
        beams at once. The decoder's incremental cache is (L, T, B*K, C), so reordering
        beams means index_select along dim=2.
        """
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        max_chars = self.params["training_params"]["max_char_prediction"]
        max_lines = self.params["training_params"].get("max_line_pred", 100)
        max_per_line = self.params["training_params"].get("max_pred_per_line", 150)
        eval_depth = self.params["training_params"].get("eval_decoder_depth", None)
        length_penalty = self.params["training_params"].get("beam_length_penalty", 1.0)

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            k = beam_size
            bk = b * k

            features = self.evaluate_encoder(x, batch_data)
            features_size = features.size()
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            # (HW, B, C) -> (HW, B*K, C); each sample's features shared by its K beams
            hw, _, c_dim = pos_features.size()
            pos_features = pos_features.unsqueeze(2).expand(hw, b, k, c_dim).reshape(hw, bk, c_dim)
            reduced_size_bk = [rs for rs in reduced_size for _ in range(k)]
            features_size_bk = (bk, features_size[1], features_size[2], features_size[3])

            start_tok = self.dataset.tokens["start"]
            end_tok = self.dataset.tokens["end"]
            pad_tok = self.dataset.tokens["pad"]

            tokens = torch.full((bk, 1), start_tok, dtype=torch.long, device=self.device)
            # Only beam 0 of each sample is live initially; the rest are -inf so the first
            # expansion cannot produce K identical hypotheses.
            scores = torch.full((b, k), float("-inf"), device=self.device)
            scores[:, 0] = 0.0
            scores = scores.view(bk)

            finished = torch.zeros(bk, dtype=torch.bool, device=self.device)
            prediction_len = torch.ones(bk, dtype=torch.long, device=self.device)
            newline_token = self.dataset.charset.index("\n") if "\n" in self.dataset.charset else -1
            line_count = torch.zeros(bk, dtype=torch.long, device=self.device)
            char_in_line = torch.zeros(bk, dtype=torch.long, device=self.device)
            cache = None

            for step in range(1, max_chars + 1):
                line_indices = index_in_lines = None
                if self.params["model_params"].get("use_line_indices", False):
                    line_indices, index_in_lines = self.get_line_indices_from_tokens(tokens)

                _, pred, cache, _ = self.models["decoder"](
                    pos_features, pos_features, tokens, reduced_size_bk,
                    prediction_len, features_size_bk, start=0, cache=cache, num_pred=1,
                    line_indices=line_indices, index_in_lines=index_in_lines,
                    padding_value=pad_tok, depth=eval_depth)

                logp = torch.log_softmax(pred[:, :, -1].float(), dim=1)   # (B*K, V)
                vocab = logp.size(1)

                # A finished beam may only extend itself with <pad> at zero cost, so it
                # keeps its score and stops competing for new tokens.
                logp[finished] = float("-inf")
                logp[finished, pad_tok] = 0.0

                cand = scores.unsqueeze(1) + logp                          # (B*K, V)
                cand = cand.view(b, k * vocab)
                top_scores, top_idx = torch.topk(cand, k, dim=1)           # (B, K)

                beam_origin = torch.div(top_idx, vocab, rounding_mode="floor")  # (B, K)
                next_token = top_idx % vocab                                    # (B, K)

                flat_origin = (beam_origin + torch.arange(b, device=self.device).unsqueeze(1) * k).view(bk)

                tokens = tokens.index_select(0, flat_origin)
                tokens = torch.cat([tokens, next_token.view(bk, 1)], dim=1)
                scores = top_scores.reshape(bk)
                finished = finished.index_select(0, flat_origin)
                line_count = line_count.index_select(0, flat_origin)
                char_in_line = char_in_line.index_select(0, flat_origin)
                prediction_len = prediction_len.index_select(0, flat_origin)
                if cache is not None:
                    cache = cache.index_select(2, flat_origin)

                new_tok = next_token.view(bk)
                just_ended = torch.eq(new_tok, end_tok) & ~finished
                if newline_token >= 0:
                    is_nl = torch.eq(new_tok, newline_token)
                    line_count = line_count + is_nl.long()
                    char_in_line = torch.where(is_nl, torch.zeros_like(char_in_line), char_in_line + 1)
                    just_ended = just_ended | ((line_count >= max_lines) | (char_in_line >= max_per_line)) & ~finished

                prediction_len = torch.where(finished, prediction_len, prediction_len + 1)
                finished = finished | just_ended
                if bool(finished.all()):
                    break

            # Length-normalised score; pick the best beam per sample.
            norm = scores / (prediction_len.float().clamp(min=1) ** length_penalty)
            best = torch.argmax(norm.view(b, k), dim=1)
            best_flat = best + torch.arange(b, device=self.device) * k

            seqs = tokens.index_select(0, best_flat)[:, 1:]        # drop <sot>
            lens = prediction_len.index_select(0, best_flat)
            pred_tokens = [seqs[i, :lens[i]] for i in range(b)]

            ind_to_remove = [end_tok, pad_tok]
            if "blank" in self.dataset.tokens:
                ind_to_remove.append(self.dataset.tokens["blank"])
            cleaned = []
            for t in pred_tokens:
                keep = torch.ones_like(t, dtype=torch.bool)
                for ind in ind_to_remove:
                    keep &= t != ind
                cleaned.append(t[keep])
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in cleaned]

        process_time = time.time() - start_time
        return {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": [[] for _ in range(b)],
            "time": process_time,
            "names": batch_data["names"],
        }

    def evaluate_batch_greedy(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        max_chars = self.params["training_params"]["max_char_prediction"]
        # Line-based limits to prevent infinite loops during early training
        max_lines = self.params["training_params"].get("max_line_pred", 100)
        max_per_line = self.params["training_params"].get("max_pred_per_line", 150)
        # None = use the full decoder stack; an int pins the inference-time depth budget
        # for an adaptive-depth model.
        eval_depth = self.params["training_params"].get("eval_decoder_depth", None)
        # Inference-time attention coverage. Accumulates how much attention mass the
        # decoder has already spent on each encoder position and subtracts a penalty from
        # future attention logits, pushing it forward through the page instead of
        # re-reading a region it already transcribed. 0 disables (default).
        cov_gain = self.params["training_params"].get("coverage_gain", 0.0) or 0.0
        cov_mode = self.params["training_params"].get("coverage_mode", "log")
        coverage = None

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b,), dtype=torch.bool, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
            prediction_len = torch.ones((b,), dtype=torch.int, device=self.device)

            # Line tracking for early stopping
            newline_token = self.dataset.charset.index("\n") if "\n" in self.dataset.charset else -1
            line_count = torch.zeros((b,), dtype=torch.int, device=self.device)
            char_in_line = torch.zeros((b,), dtype=torch.int, device=self.device)

            whole_output = list()
            confidence_scores = list()
            cache = None

            features = self.evaluate_encoder(x, batch_data)
            features_size = features.size()
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            for i in range(1, max_chars+1):
                line_indices = None
                index_in_lines = None
                if "use_line_indices" in self.params["model_params"] and self.params["model_params"]["use_line_indices"]:
                    line_indices, index_in_lines = self.get_line_indices_from_tokens(predicted_tokens)
                cov_bias = None
                if cov_gain > 0 and coverage is not None:
                    # log1p saturates, so a long document cannot build an unbounded penalty
                    # that starves every position late in decoding.
                    cov_bias = (torch.log1p(coverage) if cov_mode == "log"
                                else coverage) * cov_gain
                output, pred, cache, weights = self.models["decoder"](pos_features, pos_features, predicted_tokens,
                                                                      reduced_size,
                                                                      prediction_len, features_size, start=0,
                                                                      cache=cache, num_pred=1,
                                                                      line_indices=line_indices,
                                                                      index_in_lines=index_in_lines,
                                                                      padding_value=self.dataset.tokens["pad"],
                                                                      depth=eval_depth,
                                                                      coverage_bias=cov_bias)
                if cov_gain > 0:
                    w = weights.detach().float()
                    w = w.reshape(w.size(0), -1)      # (B, H*W), matching memory order
                    coverage = w if coverage is None else coverage + w
                whole_output.append(output)
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                new_token = torch.argmax(pred[:, :, -1], dim=1, keepdim=True)
                predicted_tokens = torch.cat([predicted_tokens, new_token], dim=1)

                # Update line tracking
                if newline_token >= 0:
                    is_newline = torch.eq(new_token.squeeze(-1), newline_token)
                    line_count = line_count + is_newline.int()
                    char_in_line = torch.where(is_newline, torch.zeros_like(char_in_line), char_in_line + 1)
                    # Check line limits
                    limit_exceeded = torch.logical_or(line_count >= max_lines, char_in_line >= max_per_line)
                    reached_end = torch.logical_or(reached_end, limit_exceeded)

                reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))

                prediction_len[torch.eq(reached_end, False)] = i + 1
                if torch.all(reached_end):
                    break

            prediction_len[torch.eq(reached_end, False)] = max_chars
            predicted_tokens = predicted_tokens[:, 1:]
            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            pred_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]

            ind_to_remove = [self.dataset.tokens["end"]]
            if "blank" in self.dataset.tokens:
                ind_to_remove.append(self.dataset.tokens["blank"])
            pred_tokens, confidence_scores = self.remove_ind_from_pred_list(pred_tokens, ind_to_remove, confidence_scores)
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in pred_tokens]

        process_time = time.time() - start_time

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
            "names": batch_data["names"]
        }

        return values
