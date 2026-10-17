"""Tokenization and causal-LM scoring primitives migrated from the supplied reference.

Inputs are lists of prompt/response strings or token tensors (B, T); outputs keep
next-token labels and response masks aligned. No CLI parameters or downloads.
Validation (macOS / Windows CMD): uv run pytest tests/test_grpo.py.
"""
from typing import Dict, List
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Tensor]:
    # 由 Codex 迁移参考并适配实现。
    """
    Tokenize prompt and output separately, concatenate them, then build:
      - input_ids: concat[:-1]
      - labels:    concat[1:]
      - response_mask: 1 on label positions corresponding to output tokens, else 0
    Shapes: (batch_size, max_len - 1)
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt_strs and output_strs must have same length, got {len(prompt_strs)} vs {len(output_strs)}"
        )

    batch_size = len(prompt_strs)
    if batch_size == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "labels": empty, "response_mask": empty}

    # pad id fallback
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # tokenize separately then concat (no special tokens)
    concat_ids_list: List[List[int]] = []
    prompt_lens: List[int] = []
    output_lens: List[int] = []

    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        o_ids = tokenizer(o, add_special_tokens=False).input_ids
        prompt_lens.append(len(p_ids))
        output_lens.append(len(o_ids))
        concat_ids_list.append(list(p_ids) + list(o_ids))

    max_len = max(1, max(len(x) for x in concat_ids_list))  # length of full concat (prompt+output)

    # full concatenated and padded (batch, max_len)
    full = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    # response_mask aligns with labels (= full[:, 1:]) => (batch, max_len - 1)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.long)

    for i, ids in enumerate(concat_ids_list):
        L = len(ids)
        full[i, :L] = torch.tensor(ids, dtype=torch.long)

        P = prompt_lens[i]
        O = output_lens[i]

        # output tokens are full indices: [P, P+O-1]
        # labels are full shifted left => label indices: [P-1, P+O-2] (length O)
        if O > 0:
            start = max(P - 1, 0)
            end = P + O - 1  # cap to mask length
            response_mask[i, start:end] = 1

    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # 由 Codex 迁移参考并适配实现。
    """
    Compute per-token entropy of next-token distribution (over vocab dim).

    Args:
        logits: (batch_size, sequence_length, vocab_size)

    Returns:
        entropies: (batch_size, sequence_length)
    """
    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dim, got {tuple(logits.shape)}")

    # log_probs = logits - logsumexp(logits)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)   # (B, T, 1)
    log_probs = logits - log_z                              # (B, T, V)
    probs = torch.exp(log_probs)                            # (B, T, V)

    # H(p) = - sum_v p(v) * log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)              # (B, T)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    # 由 Codex 迁移参考并适配实现。
    """
    Compute per-token conditional log-probabilities for a causal LM.

    Args:
        model: HF causal LM, already on correct device.
        input_ids: (B, T)
        labels:    (B, T) token ids
        return_token_entropy: if True, also return per-token entropy (B, T) computed from logits.

    Returns:
        dict with:
          - "log_probs": (B, T)
          - "token_entropy": (B, T) if requested
    """
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            f"input_ids and labels must be (B, T). Got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have same shape. Got {tuple(input_ids.shape)} vs {tuple(labels.shape)}"
        )

    # forward
    logits = model(input_ids=input_ids).logits  # (B, T, V)

    # stable log-probs over vocab
    log_probs_vocab = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # gather log p(label_t | prefix) for each position
    # labels: (B, T) -> (B, T, 1) for gather
    gathered = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1))  # (B, T, 1)
    token_log_probs = gathered.squeeze(-1)  # (B, T)

    out: Dict[str, torch.Tensor] = {"log_probs": token_log_probs}

    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)  # (B, T)

    return out
