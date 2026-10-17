"""由 Codex 从用户指定参考目录迁移：完成作业要求的实现；保留参考算法。"""

import math
import torch


def flash_bwd_recompute_impl(q, k, v, o, do, L, is_causal=False):
    """由 Codex 适配参考反向公式：分块重算概率，不分配完整 Q×K 得分矩阵。"""
    scale = q.shape[-1] ** -0.5
    qf, kf, vf, dof = (x.float() for x in (q, k, v, do))
    dq, dk, dv = (torch.zeros_like(x) for x in (qf, kf, vf))
    correction = (dof * o.float()).sum(-1)
    tile = 32
    for i in range(0, q.shape[1], tile):
        qi, doi = qf[:, i : i + tile], dof[:, i : i + tile]
        for j in range(0, k.shape[1], tile):
            kj, vj = kf[:, j : j + tile], vf[:, j : j + tile]
            scores = qi @ kj.transpose(-1, -2) * scale
            if is_causal:
                allowed = torch.arange(i, i + qi.shape[1], device=q.device)[:, None] >= torch.arange(j, j + kj.shape[1], device=q.device)[None, :]
                scores = scores.masked_fill(~allowed, -1e6)
            probabilities = torch.exp(scores - L[:, i : i + tile, None].float())
            dv[:, j : j + tile] += probabilities.transpose(-1, -2) @ doi
            ds = probabilities * (doi @ vj.transpose(-1, -2) - correction[:, i : i + tile, None]) * scale
            dq[:, i : i + tile] += ds @ kj
            dk[:, j : j + tile] += ds.transpose(-1, -2) @ qi
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        # q: (B, Q, D), k/v: (B, K, D)
        if q.dim() != 3:
            raise ValueError("q must have shape (batch, queries, head_dim)")
        if k.dim() != q.dim() or v.dim() != q.dim():
            raise ValueError("q/k/v must have same rank")
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ValueError("leading dims of q/k/v must match")
        if k.shape[-2] != v.shape[-2] or k.shape[-1] != v.shape[-1]:
            raise ValueError("k and v must have sanme shape in last 2 dims")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k must have same D")

        B, Q, D = q.shape
        K = k.shape[-2]

        scale = 1.0 / math.sqrt(D)

        # choose tile sizes (>=16)
        Bq = 32
        Bk = 32

        # output buffers
        o = torch.empty((B, Q, D), device=q.device, dtype=q.dtype)
        L = torch.empty((B, Q), device=q.device, dtype=torch.float32)

        for i in range(0, Q, Bq):
            q_i = q[:, i : i + Bq, :]  # (B, Bq, D)

            # running stats for this query tile
            m = torch.full((B, q_i.shape[1]), -float("inf"), device=q.device, dtype=torch.float32)
            l = torch.zeros((B, q_i.shape[1]), device=q.device, dtype=torch.float32)
            o_acc = torch.zeros((B, q_i.shape[1], D), device=q.device, dtype=torch.float32)

            for j in range(0, K, Bk):
                k_j = k[:, j : j + Bk, :]  # (B, Bk, D)
                v_j = v[:, j : j + Bk, :]  # (B, Bk, D)

                # S = q @ k^T * scale -> (B, Bq, Bk)
                S = torch.matmul(q_i.float(), k_j.float().transpose(-1, -2)) * scale

                # 由 Codex 补全：块内使用全局位置施加因果掩码，兼容末尾不足整块的长度。
                if is_causal:
                    qi = torch.arange(i, i + q_i.shape[1], device=q.device)[:, None]
                    kj = torch.arange(j, j + k_j.shape[1], device=q.device)[None, :]
                    S = S.masked_fill(qi < kj, -1e6)
                # online softmax update
                m_new = torch.maximum(m, S.max(dim=-1).values)  # (B, Bq)

                # exp(S - m_new)
                P_tilde = torch.exp(S - m_new.unsqueeze(-1))  # (B, Bq, Bk)

                # l_new = exp(m - m_new) * l + rowsum(P_tilde)
                l_new = torch.exp(m - m_new) * l + P_tilde.sum(dim=-1)  # (B, Bq)

                # o_acc = exp(m - m_new) * o_acc + P_tilde @ v_j
                o_acc = (torch.exp(m - m_new)).unsqueeze(-1) * o_acc + torch.matmul(P_tilde, v_j.float())

                m, l = m_new, l_new

            # finalize: O = o_acc / l
            o_i = o_acc / l.unsqueeze(-1)
            o[:, i : i + Bq, :] = o_i.to(dtype=q.dtype)

            # L = logsumexp(S_row) = m + log(l)
            L[:, i : i + Bq] = m + torch.log(l)

        # save tensor for later backward stage
        ctx.save_for_backward(L, q, k, v, o)
        ctx.is_causal = is_causal

        # return output
        return o

    @staticmethod
    def backward(ctx, do):
        (L, q, k, v, o) = ctx.saved_tensors
        dq, dk, dv = flash_bwd_recompute_impl(q, k, v, o, do, L, ctx.is_causal)
        return dq, dk, dv, None
