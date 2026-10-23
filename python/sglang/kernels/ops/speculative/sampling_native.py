"""Pure-torch references for EAGLE sampling verification.

They mirror the device kernels step for step: the per-request walk runs on the
host in float32 (``numpy.float32``) so accept / reject decisions round like the
kernels do, and the vocabulary-wide residual sampling runs on the device.
"""

from __future__ import annotations

import numpy as np
import torch

_f32 = np.float32


def sample_residual(
    *, residual: torch.Tensor, coins: torch.Tensor, norm_sums: torch.Tensor
) -> torch.Tensor:
    """First index whose inclusive prefix sum exceeds ``coin * norm_sum``, per row.

    Rows with no such index take their last positive index, else ``vocab - 1``.
    """
    vocab = residual.shape[-1]
    over = (torch.cumsum(residual, dim=-1) > (coins * norm_sums).unsqueeze(1)) & (
        residual > 0
    )
    positive = residual > 0
    idx = torch.arange(vocab, device=residual.device)
    first_over = torch.where(over, idx, vocab).min(dim=-1).values
    last_positive = torch.where(positive, idx, -1).max(dim=-1).values
    fallback = torch.where(last_positive >= 0, last_positive, vocab - 1)
    return torch.where(first_over < vocab, first_over, fallback)


def tree_speculative_sampling_target_only_native(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
) -> None:
    bs, num_draft_tokens = candidates.shape
    num_spec_tokens = accept_index.shape[1]
    vocab = target_probs.shape[-1]
    target_rows = target_probs.view(bs * num_draft_tokens, vocab)
    draft_rows = draft_probs.view(bs * num_draft_tokens, vocab)
    # target_at[b][r][c]: probability, at tree node r, of node c's draft token.
    target_at = (
        target_probs.view(bs, num_draft_tokens, vocab)
        .gather(2, candidates.unsqueeze(1).expand(bs, num_draft_tokens, -1))
        .tolist()
    )
    cand = candidates.tolist()
    index = retrive_index.tolist()
    next_token = retrive_next_token.tolist()
    next_sibling = retrive_next_sibling.tolist()
    coins = uniform_samples.tolist()
    threshold_single = _f32(threshold_single)
    threshold_acc = _f32(max(threshold_acc, 1e-9))

    predict_at, predict_val, accept_at, accept_val = [], [], [], []
    rejected_rows, rejected_tokens, final_rows, final_last, counts = [], [], [], [], []
    for b in range(bs):
        prob_acc = _f32(0.0)
        cur_node = 0  # node whose target distribution scores the children
        coin = _f32(coins[b][0])
        last = index[b][0]
        accept_at.append(b * num_spec_tokens)
        accept_val.append(last)
        num_accept = 0
        cur = 0
        for _ in range(1, num_spec_tokens):
            cur = next_token[b][cur]
            while cur != -1:
                p = _f32(target_at[b][cur_node][cur])
                prob_acc = _f32(prob_acc + p)
                if p > 0 and (coin < prob_acc / threshold_acc or p >= threshold_single):
                    prob_acc = _f32(0.0)
                    cur_node = cur
                    coin = _f32(coins[b][cur])
                    predict_at.append(last)
                    predict_val.append(cand[b][cur])
                    num_accept += 1
                    last = index[b][cur]
                    accept_at.append(b * num_spec_tokens + num_accept)
                    accept_val.append(last)
                    break
                # The residual below must not resample a rejected token.
                rejected_rows.append(b * num_draft_tokens + cur_node)
                rejected_tokens.append(cand[b][cur])
                cur = next_sibling[b][cur]
            if cur == -1:
                break
        counts.append(num_accept)
        final_rows.append(b * num_draft_tokens + cur_node)
        final_last.append(last)

    device = target_probs.device
    if rejected_rows:
        rows = torch.tensor(rejected_rows, device=device)
        toks = torch.tensor(rejected_tokens, device=device)
        draft_rows[rows, toks] = target_rows[rows, toks]
    rows = torch.tensor(final_rows, device=device)
    # The bonus position (every draft accepted) has no draft distribution.
    has_draft = torch.tensor(counts, device=device) != num_spec_tokens - 1
    residual = (target_rows[rows] - draft_rows[rows] * has_draft.unsqueeze(1)).clamp(
        min=0
    )
    sampled = sample_residual(
        residual=residual,
        coins=uniform_samples_for_final_sampling,
        norm_sums=residual.sum(dim=-1),
    )
    _scatter(predicts, predict_at + final_last, predict_val, extra=sampled)
    _scatter(accept_index, accept_at, accept_val)
    accept_token_num.copy_(torch.tensor(counts, dtype=accept_token_num.dtype))


def chain_speculative_sampling_native(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    block_verification: bool,
) -> None:
    bs, num_slots = candidates.shape
    rows = torch.arange(bs, device=candidates.device)
    steps = torch.arange(num_slots - 1, device=candidates.device)
    draft_tokens = candidates[:, 1:]
    # p_at / q_at[b][s]: target / draft probability of draft token s + 1 at step s.
    p_at = target_probs[rows[:, None], steps[None, :], draft_tokens].tolist()
    q_at = draft_probs[rows[:, None], steps[None, :], draft_tokens].tolist()
    cand = candidates.tolist()
    index = retrive_index.tolist()
    coins = uniform_samples.tolist()

    predict_at, predict_val, accept_at, accept_val = [], [], [], []
    counts, scales, norm_sums, final_last = [], [], [], []
    for b in range(bs):
        last = index[b][0]
        accept_at.append(b * num_slots)
        accept_val.append(last)
        num_accept, scale, norm_sum = 0, _f32(1.0), -1.0
        if block_verification:
            # Algorithm 2 of https://arxiv.org/abs/2403.10444.
            prefix, prefixes = _f32(1.0), []
            for s in range(num_slots - 1):
                p, q = _f32(p_at[b][s]), _f32(q_at[b][s])
                prefix = min(_f32(prefix * p / q), _f32(1.0)) if 0 < q <= 1 else _f32(0)
                prefixes.append(prefix)
            for step in range(num_slots - 1, 0, -1):
                prefix, coin = prefixes[step - 1], _f32(coins[b][step - 1])
                if coin >= prefix:
                    continue
                accept_prob, residual_mass = prefix, -1.0
                if step < num_slots - 1 and 0 < prefix < 1:
                    residual_mass = _f32(
                        (prefix * target_probs[b, step] - draft_probs[b, step])
                        .clamp(min=0)
                        .sum()
                        .item()
                    )
                    accept_prob = residual_mass / (residual_mass + 1 - prefix)
                if coin < accept_prob:
                    num_accept, scale, norm_sum = step, prefix, residual_mass
                    break
            accepted = range(1, num_accept + 1)
        else:
            accepted = []
            for step in range(1, num_slots):
                p, q = _f32(p_at[b][step - 1]), _f32(q_at[b][step - 1])
                if not (0 < q <= 1 and _f32(coins[b][step - 1]) * q < p):
                    break
                accepted.append(step)
            num_accept = len(accepted)
        for step in accepted:
            predict_at.append(last)
            predict_val.append(cand[b][step])
            last = index[b][step]
            accept_at.append(b * num_slots + step)
            accept_val.append(last)
        counts.append(num_accept)
        scales.append(float(scale))
        norm_sums.append(float(norm_sum))
        final_last.append(last)

    device = target_probs.device
    count_t = torch.tensor(counts, device=device)
    all_accepted = count_t == num_slots - 1
    p_rows = target_probs[rows, count_t]
    # draft_probs has num_slots - 1 rows; the all-accepted rows never read it.
    q_rows = draft_probs[rows, count_t.clamp(max=draft_probs.shape[1] - 1)]
    q_rows = torch.where((q_rows >= 0) & (q_rows <= 1), q_rows, 0.0)
    scale_t = torch.tensor(scales, device=device, dtype=p_rows.dtype).unsqueeze(1)
    residual = torch.where(
        all_accepted.unsqueeze(1), p_rows, (scale_t * p_rows - q_rows).clamp(min=0)
    )
    norm_t = torch.tensor(norm_sums, device=device, dtype=p_rows.dtype)
    norm_t = torch.where(norm_t < 0, residual.sum(dim=-1), norm_t)
    target_u = uniform_samples_for_final_sampling * norm_t
    over = torch.cumsum(residual, dim=-1) > target_u.unsqueeze(1)
    vocab = residual.shape[-1]
    idx = torch.arange(vocab, device=device)
    first_over = torch.where(over, idx, vocab).min(dim=-1).values
    sampled = torch.where(first_over < vocab, first_over, vocab - 1)
    _scatter(predicts, predict_at + final_last, predict_val, extra=sampled)
    _scatter(accept_index, accept_at, accept_val)
    accept_token_num.copy_(torch.tensor(counts, dtype=accept_token_num.dtype))


def _scatter(
    out: torch.Tensor,
    offsets: list,
    values: list,
    *,
    extra: torch.Tensor | None = None,
) -> None:
    """``out.view(-1)[offsets] = values (+ extra)``, with ``extra`` already on device."""
    vals = torch.tensor(values, dtype=out.dtype, device=out.device)
    if extra is not None:
        vals = torch.cat([vals, extra.to(out.dtype)])
    out.view(-1)[torch.tensor(offsets, dtype=torch.long, device=out.device)] = vals
