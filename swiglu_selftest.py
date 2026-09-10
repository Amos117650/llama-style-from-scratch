# swiglu_selftest.py — SwiGLU FFN 自测裁判（四关）
# 用法：把你的 SwiGLU 类和 silu 粘到下方粘贴区，然后
#   E:\anaconda3\envs\cs336\python.exe swiglu_selftest.py
# 接口约定：
#   class SwiGLU(nn.Module): __init__(self, dim=192, hidden=512)
#     三个 nn.Linear 全部 bias=False，命名 w1/w2/w3（或 HF 风格 gate_proj/up_proj/down_proj）
#     forward(x): [B,T,dim] -> [B,T,dim]，实现 w2( silu(w1 x) ⊙ w3 x )
#   def silu(x): 一行实现（默写要求：不用 F.silu）
# 四关各抓什么：
#   1) silu 数学关：过原点 + 负半轴非单调 + 与 F.silu 逐点一致
#   2) 门控语义关：w3 支路整体 ×c => 输出精确 ×c（乘性门控的齐次性签名）+ 门全关测试
#   3) 交叉验证关：独立写法 F.linear 逐步手算对照（抓：漏 silu / silu 放错支路 / w1,w3 写反）
#   4) 参数对账关：FFN 参数 = 3×192×512 = 8×192² = 294,912，bias 全 None，hidden=512

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ============ 粘贴区 ============
class SwiGLU(nn.Module):
    def __init__(self, n_embed=192):
        super().__init__()
        hidden = 2*4*n_embed // 3
        self.gate_proj = nn.Linear(n_embed, hidden, bias=False) # w1(门支路，silu 作用在这)
        self.up_proj = nn.Linear(n_embed, hidden, bias=False)   # w3(值支路，纯线性)
        self.down_proj = nn.Linear(hidden, n_embed, bias=False) # w2(投影回 d)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * silu(self.gate_proj(x)))

def silu(x):
    return x * torch.sigmoid(x)
    raise NotImplementedError
# ============ 粘贴区结束 ============


def _linears(module):
    """按 w1/w2/w3（或 HF 风格）取三个 Linear，统一返回 (gate, down, up) = (w1, w2, w3)"""
    for g, d, u in (("w1", "w2", "w3"), ("gate_proj", "down_proj", "up_proj")):
        if all(hasattr(module, n) for n in (g, d, u)):
            return getattr(module, g), getattr(module, d), getattr(module, u)
    raise AttributeError(
        "找不到 w1/w2/w3（或 gate_proj/up_proj/down_proj）三个 nn.Linear——命名先对齐再谈别的"
    )


def test_silu():
    x = torch.linspace(-8, 8, 2001)
    assert torch.allclose(silu(x), F.silu(x), atol=1e-6), "silu 与 F.silu 不一致——公式是 x·sigmoid(x)"
    assert abs(silu(torch.tensor(0.0)).item()) < 1e-8, "silu(0) 应为 0（过原点）"
    lo, l, r = silu(torch.tensor(-1.28)), silu(torch.tensor(-3.0)), silu(torch.tensor(-1.0))
    assert lo < l and lo < r, "silu 负半轴应有内部最低点（先降后升，非单调）——纯 ReLU 没有这个"
    print("[PASS] 1/4 silu 数学：过原点 + 负半轴非单调 + 与 F.silu 逐点一致")


def test_gating_signature():
    m = SwiGLU()
    m.eval()
    x = torch.randn(2, 16, 192)
    with torch.no_grad():
        out1 = m(x)
        m2 = copy.deepcopy(m)
        _linears(m2)[2].weight *= 3.7  # 线性支路 w3 整体放大 c 倍
        out2 = m2(x)
    # 门控乘积 + w2 无 bias => 输出精确放大 c 倍。若 silu 跑到了 w3 支路上，这里必挂。
    assert torch.allclose(out2, 3.7 * out1, rtol=1e-4, atol=1e-5), \
        "w3 支路 ×3.7 后输出没有精确 ×3.7——乘积结构或 silu 的位置不对"
    # 门全关：输入全 1、w1 置全 -1 => w1x = -192 => silu 下溢 ≈ 0 => 输出应归零
    m3 = copy.deepcopy(m)
    with torch.no_grad():
        _linears(m3)[0].weight.fill_(-1.0)
        out3 = m3(torch.ones(1, 5, 192))
    assert out3.abs().max() < 1e-20, \
        f"门应全关（w1x=-192 时 silu≈0），但输出 max={out3.abs().max():.2e}——门控没起作用或漏了 silu"
    print("[PASS] 2/4 门控语义：输出随 w3 支路精确线性缩放 + 门全关时输出归零")


def test_crosscheck():
    m = SwiGLU()
    m.eval()
    g, d, u = _linears(m)
    assert tuple(g.weight.shape) == (512, 192), \
        f"w1 形状 {tuple(g.weight.shape)}，应为 (512,192)——hidden 是不是没取 ⌊8/3·192⌋=512？"
    x = torch.randn(3, 12, 192)
    with torch.no_grad():
        out = m(x)
        expected = F.linear(F.silu(F.linear(x, g.weight)) * F.linear(x, u.weight), d.weight)
    assert torch.allclose(out, expected, atol=1e-4), \
        "与独立手算 w2(silu(w1x)·w3x) 不一致——常见原因：漏 silu / silu 放到了 w3 支路 / w1 与 w3 写反"
    print("[PASS] 3/4 交叉验证：forward 与 F.linear 独立手算逐点一致")


def test_param_accounting():
    m = SwiGLU()
    g, d, u = _linears(m)
    for name, lin in zip(("w1/gate", "w2/down", "w3/up"), (g, d, u)):
        assert lin.bias is None, f"{name} 带 bias——规格要求三个 Linear 全部 bias=False（LLaMA 风格）"
    n = sum(p.numel() for p in m.parameters())
    target = 3 * 192 * 512
    assert n == target == 8 * 192 * 192, \
        f"FFN 参数 {n}，应为 {target}（=8×192²）——检查 hidden、多余层或漏层"
    print(f"[PASS] 4/4 参数对账：{n:,} = 3×192×512 = 8×192²（与 GPT-2 双矩阵 2×192×768 严格等量）")


if __name__ == "__main__":
    test_silu()
    test_gating_signature()
    test_crosscheck()
    test_param_accounting()
    print("\n全绿：你的 SwiGLU 数学上是对的，可以接进 testv4.py 训练了。")
    print("没全绿也不用慌——每个 FAIL 都精确指向一个 bug，修完重跑。")
