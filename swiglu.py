import torch
import torch.nn as nn
from torch.nn import functional as F


class SwiGLU(torch.nn.Module):
    def __init__(self, n_embed=192):
        super().__init__()
        hidden=2*4*n_embed//3
        self.gate_proj = nn.Linear(n_embed, hidden, bias=False) # w1(门支路，silu 作用在这)
        self.up_proj = nn.Linear(n_embed, hidden, bias=False)   # w3(值支路，纯线性)
        self.down_proj = nn.Linear(hidden, n_embed, bias=False) # w2(投影回 d)
        # 你的三个 nn.Linear 和 forward
        # raise NotImplementedError有什么用?
        # raise NotImplementedError 是一个占位符，用于提醒开发者在此处实现具体的功能。
        # 在这个上下文中，它表示你需要在这里实现 SwiGLU 的 forward 方法和三个 nn.Linear 的初始化。
        # 如果你直接运行代码而没有实现这些部分，程序会抛出 NotImplementedError 异常，提示你还没有完成这部分的实现。

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * self.silu(self.gate_proj(x)))

    def silu(self, x):
        return x * torch.sigmoid(x)