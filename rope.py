import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, theta, head_size, max_seq_len):
        super().__init__()
        # inv_freq是一个一维张量，表示每个位置的逆频率向量
        inv_freq = theta ** (-torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)   # (head_size/2,)
        # pos是一个一维张量，表示位置索引，从0到max_seq_len-1。
        # freqs是一个二维张量，表示每个位置的频率向量
        pos = torch.arange(max_seq_len, dtype=torch.float32)    # (max_seq_len,)
        # torch.outer(pos, inv_freq)计算pos和inv_freq的外积，得到一个二维张量，其中每一行对应一个位置，每一列对应一个频率。
        # freqs有什么用,它的每个元素代表什么?
        # freqs的每个元素表示在给定位置和频率下的旋转角度。
        # 具体来说，freqs[i, j]表示位置i和频率j对应的旋转角度。
        # 这个旋转角度用于计算旋转嵌入，从而在注意力机制中引入位置信息。
        freqs = torch.outer(pos, inv_freq)
        # 為什么要额外加上cos(),sin()?
        # 这是因为旋转嵌入的计算需要使用余弦和正弦函数来生成旋转矩阵。
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x, pos_ids=None):
        B, T, C = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(T, device=x.device)

        cos = self.cos[pos_ids]
        sin = self.sin[pos_ids]
        # 为什么要这样写x1, x2?
        # 这是因为旋转嵌入的计算需要将输入张量x分成两部分，分别对应于余弦和正弦的旋转。
        # x[..., :C//2]表示取x的最后一个维度的前一半，x[..., C//2:]表示取x的最后一个维度的后一半。
        x1, x2 = x[..., :C//2], x[..., C//2:]   # x1: (B, T, C//2), x2: (B, T, C//2)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)