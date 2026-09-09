import torch
import torch.nn as nn
import torch.nn.functional as F


class RopeEmbedding(nn.Module):
    def __init__(self, theta, head_size, max_seq_len):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer('cos',freqs.cos())
        self.register_buffer('sin',freqs.sin())

    def forward(self, x, pos_ids=None):
        B, T, C = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(T, device=x.device)

        cos = self.cos[pos_ids]
        sin = self.sin[pos_ids]
        x1, x2 = x[..., :C//2], x[..., C//2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)



if __name__ == "__main__":
    r = RopeEmbedding(10000.0, 32, 128)
    q, k = torch.randn(1, 8, 32), torch.randn(1, 8, 32)
    A = lambda x, off=0: r(x, pos_ids=torch.arange(off, off + x.shape[-2]))
    s0 = (A(q) @ A(k).transpose(-2, -1)).squeeze()
    s5 = (A(q, 5) @ A(k, 5).transpose(-2, -1)).squeeze()
    print("平移不变性 diff:", (s0 - s5).abs().max().item())   # 期望 ~1e-6