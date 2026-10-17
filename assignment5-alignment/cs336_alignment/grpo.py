"""2026 GRPO primitives and one optimizer update, adapted by Codex.

All log-probability/loss tensors have shape (B, T), rewards have shape (B,),
and masks select response tokens. No model loading or rollout generation occurs.
Validation on macOS / Windows CMD: uv run pytest tests/test_grpo.py.
"""
from typing import Callable, Literal

import torch

from .sft_utils import get_response_log_probs, tokenize_prompt_and_output


def compute_rollout_rewards(reward_fn: Callable, rollout_responses: list[str],
                            repeated_ground_truths: list[str]):
    """Return reward vector (B,) and means of each grader component."""
    # 由 Codex 根据目标文档实现；参考版将评分与组归一化耦合，此处拆分。
    if len(rollout_responses) != len(repeated_ground_truths) or not rollout_responses:
        raise ValueError('Responses and ground truths must have equal, nonzero length')
    results = [reward_fn(response, truth) for response, truth in
               zip(rollout_responses, repeated_ground_truths)]
    rewards = torch.tensor([result['reward'] for result in results], dtype=torch.float32)
    metadata = {key: sum(result[key] for result in results) / len(results)
                for key in results[0]}
    return rewards, metadata


def compute_group_normalized_rewards(raw_rewards: torch.Tensor, group_size: int,
                                     baseline: Literal['mean', 'none'] = 'mean',
                                     advantage_eps: float = 1e-6,
                                     advantage_normalizer: Literal['std', 'none', 'mean'] = 'std'):
    """Compute group-relative advantages (B,), including Dr.GRPO/RFT/MaxRL variants."""
    # 由 Codex 参考旧版组归一化并按 2026 接口修复并实现。
    if raw_rewards.ndim != 1 or group_size < 1 or raw_rewards.numel() % group_size:
        raise ValueError('Rewards must be a vector whose length is divisible by group_size')
    grouped = raw_rewards.detach().reshape(-1, group_size)
    means = grouped.mean(dim=-1, keepdim=True)
    if baseline == 'mean':
        advantages = grouped - means
    elif baseline == 'none':
        advantages = grouped.clone()
    else:
        raise ValueError(f'Unsupported baseline: {baseline}')
    if advantage_normalizer == 'std':
        # The handout explicitly requests torch.std's default sample correction.
        normalizer = grouped.std(dim=-1, keepdim=True) if group_size > 1 else torch.zeros_like(means)
    elif advantage_normalizer == 'mean':
        normalizer = means
    elif advantage_normalizer == 'none':
        return advantages.flatten(), {'reward_mean': raw_rewards.mean().item()}
    else:
        raise ValueError(f'Unsupported advantage normalizer: {advantage_normalizer}')
    return (advantages / (normalizer + advantage_eps)).flatten(), {'reward_mean': raw_rewards.mean().item()}


def compute_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor,
                                 policy_log_probs: torch.Tensor,
                                 importance_reweighting_method: str = 'none',
                                 old_log_probs: torch.Tensor | None = None,
                                 cliprange: float | None = None,
                                 response_mask: torch.Tensor | None = None):
    """Return per-token negative policy objective (B,T) and clipping diagnostics."""
    # 由 Codex 参考旧版 GRPO loss 并补全 GSPO 修复并实现。
    advantages = raw_rewards_or_advantages.detach().reshape(-1, 1)
    if importance_reweighting_method == 'none':
        return -advantages * policy_log_probs, {}
    if importance_reweighting_method not in {'noclip', 'grpo', 'gspo'}:
        raise ValueError(f'Unsupported importance method: {importance_reweighting_method}')
    if old_log_probs is None or old_log_probs.shape != policy_log_probs.shape:
        raise ValueError('Matching old_log_probs are required for off-policy updates')
    log_ratio = policy_log_probs - old_log_probs.detach()
    if importance_reweighting_method == 'gspo':
        if response_mask is None:
            raise ValueError('GSPO requires a response_mask')
        # Geometric mean ratio uses only response tokens, then broadcasts to T.
        log_ratio = ((log_ratio * response_mask).sum(dim=-1, keepdim=True) /
                     response_mask.sum(dim=-1, keepdim=True).clamp_min(1))
    ratio = log_ratio.exp()
    objective = ratio * advantages
    if importance_reweighting_method == 'noclip':
        return -objective, {}
    if cliprange is None or cliprange < 0:
        raise ValueError('A nonnegative cliprange is required for clipping')
    clipped_objective = ratio.clamp(1 - cliprange, 1 + cliprange) * advantages
    loss = -torch.minimum(objective, clipped_objective)
    if importance_reweighting_method == 'gspo':
        loss = loss.expand_as(policy_log_probs)
    return loss, {'clip_fraction': (clipped_objective < objective).float().detach()}


def aggregate_loss_across_microbatch(per_token_policy_gradient_loss: torch.Tensor,
                                     mask: torch.Tensor,
                                     loss_normalization: str = 'sequence',
                                     normalization_constant: int | None = None):
    """Reduce (B,T) losses to a scalar using sequence or fixed-token normalization."""
    # 由 Codex 根据目标 handout 修复并实现。
    masked_sums = (per_token_policy_gradient_loss * mask).sum(dim=-1)
    if loss_normalization == 'sequence':
        return (masked_sums / mask.sum(dim=-1).clamp_min(1)).mean()
    if loss_normalization == 'constant':
        if normalization_constant is None or normalization_constant <= 0:
            raise ValueError('constant normalization requires a positive normalization_constant')
        return masked_sums.sum() / normalization_constant
    raise ValueError(f'Unsupported loss normalization: {loss_normalization}')


def grpo_train_step(model, tokenizer, optimizer, gradient_accumulation_steps: int,
                    max_grad_norm: float | None, reward_fn: Callable,
                    repeated_prompts: list[str], rollout_responses: list[str],
                    repeated_ground_truths: list[str], group_size: int,
                    baseline: str = 'mean', advantage_eps: float = 1e-6,
                    advantage_normalizer: str = 'std',
                    importance_reweighting_method: str = 'none',
                    old_log_probs: torch.Tensor | None = None,
                    cliprange: float | None = None,
                    loss_normalization: str = 'sequence',
                    normalization_constant: int | None = None):
    """Accumulate rollout gradients and update once; skip zero-advantage rows.

    Microbatch losses retain their weight in the original batch even after pruning.
    The return loss is detached; optimizer gradients are reset after the update.
    """
    # 由 Codex 根据 2026 目标接口修复并实现。
    batch_size = len(repeated_prompts)
    if batch_size == 0 or len(rollout_responses) != batch_size:
        raise ValueError('Prompt and response lists must have equal, nonzero length')
    if not 1 <= gradient_accumulation_steps <= batch_size:
        raise ValueError('gradient_accumulation_steps must be between 1 and batch size')
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in
               tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer).items()}
    rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, _ = compute_group_normalized_rewards(rewards, group_size, baseline,
                                                    advantage_eps, advantage_normalizer)
    advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    token_count = torch.zeros((), device=device)
    clipped_count = torch.zeros((), device=device)
    metadata = dict(reward_metadata)
    for indices in torch.tensor_split(torch.arange(batch_size, device=device), gradient_accumulation_steps):
        indices = indices[advantages[indices] != 0]
        if not indices.numel():
            continue
        scores = get_response_log_probs(model, encoded['input_ids'][indices],
                                        encoded['labels'][indices], return_token_entropy=True)
        mask = encoded['response_mask'][indices]
        losses, loss_metadata = compute_policy_gradient_loss(
            advantages[indices], scores['log_probs'], importance_reweighting_method,
            old_log_probs[indices] if old_log_probs is not None else None, cliprange, mask)
        loss = aggregate_loss_across_microbatch(losses, mask, loss_normalization,
                                                normalization_constant)
        if loss_normalization == 'sequence':
            loss = loss * (indices.numel() / batch_size)
        loss.backward()
        total_loss += loss.detach()
        total_entropy += (scores['token_entropy'].detach() * mask).sum()
        token_count += mask.sum()
        if 'clip_fraction' in loss_metadata:
            clipped_count += (loss_metadata['clip_fraction'] * mask).sum()
    if max_grad_norm is not None:
        metadata['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata['token_entropy'] = (total_entropy / token_count.clamp_min(1)).item()
    if importance_reweighting_method in {'grpo', 'gspo'}:
        metadata['clip_fraction'] = (clipped_count / token_count.clamp_min(1)).item()
    return total_loss, metadata
