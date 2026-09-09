# rope_selftest.py — RoPE 自测裁判
# 用法：把你的实现粘贴到下方标记区，然后 python rope_selftest.py
# 接口约定（如果你的实现是 complex 风格 freqs_cis，写个 2 行适配器转成 cos/sin 即可）：
#   precompute_freqs_cis(dim, seq_len, theta=10000.0) -> (cos, sin)  各 shape [seq_len, dim]，fp32
#   apply_rotary_emb(x, cos, sin) -> 与 x 同形状（x: [batch, n_heads, seq_len, head_dim]）

import torch

torch.manual_seed(42)

# ============ 粘贴区：把你的三个函数放进来 ============
def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    raise NotImplementedError

def rotate_half(x):
    raise NotImplementedError

def apply_rotary_emb(x, cos, sin):
    raise NotImplementedError
# ============ 粘贴区结束 ============


def test_shapes():
    q = torch.randn(2, 6, 10, 32)
    cos, sin = precompute_freqs_cis(32, 10)
    out = apply_rotary_emb(q, cos, sin)
    assert out.shape == q.shape, f"shape 变了: {q.shape} -> {out.shape}"
    assert out.dtype == q.dtype
    # 不应改变范数（旋转保长度）：每个 token 向量的 2-范数不变
    n_before = q.norm(dim=-1)
    n_after = out.norm(dim=-1)
    assert torch.allclose(n_before, n_after, atol=1e-5), "旋转不该改变向量范数，检查实现"
    print("[PASS] 1/3 形状保持 + 旋转保范数")


def test_translation_invariance():
    # 同一段 token 序列整体右移 s 位后，任意两 token 的 q·k 内积应完全不变（相对位置编码的签名）
    T, s = 16, 7
    q = torch.randn(1, 1, T, 32)
    k = torch.randn(1, 1, T, 32)
    cos_full, sin_full = precompute_freqs_cis(32, T + s)
    qr = apply_rotary_emb(q, cos_full[:T], sin_full[:T])        # 位置 0..T-1
    kr = apply_rotary_emb(k, cos_full[:T], sin_full[:T])
    qs = apply_rotary_emb(q, cos_full[s:], sin_full[s:])        # 位置 s..s+T-1
    ks = apply_rotary_emb(k, cos_full[s:], sin_full[s:])
    score_before = (qr @ kr.transpose(-2, -1)).squeeze()
    score_after = (qs @ ks.transpose(-2, -1)).squeeze()
    diff = (score_before - score_after).abs().max().item()
    assert diff < 1e-4, f"平移后注意力分数变了 (max diff={diff:.2e})——绝对位置泄漏进来了"
    print(f"[PASS] 2/3 平移不变性 (max diff={diff:.2e})：q·k 只依赖相对位置")


def test_toeplitz():
    # 相对位置编码 => 打分矩阵是 Toeplitz：同一条对角线内部恒定（不同对角线之间数值不同是正常的）
    # 关键：必须用「同一个 q/k 向量放到所有位置」来考——若每个位置用不同随机向量，该性质根本不成立
    T = 20
    q0, k0 = torch.randn(32), torch.randn(32)
    q, k = q0.expand(1, 1, T, 32), k0.expand(1, 1, T, 32)
    cos, sin = precompute_freqs_cis(32, T)
    score = (apply_rotary_emb(q, cos, sin) @ apply_rotary_emb(k, cos, sin).transpose(-2, -1)).squeeze()
    worst = 0.0
    for off in range(-T + 1, T):
        d = torch.diagonal(score, offset=off)
        worst = max(worst, (d - d.mean()).abs().max().item())
    assert worst < 1e-4, f"对角线内部不恒定 (max 波动={worst:.2e})——score 不只依赖 i-j"
    print(f"[PASS] 3/3 Toeplitz：每条对角线内部恒定 (max 波动 {worst:.2e})，score 只由相对位置决定")


if __name__ == "__main__":
    test_shapes()
    test_translation_invariance()
    test_toeplitz()
    print("\n全绿：你的 RoPE 数学上是对的，可以接进模型训练了。")
    print("没全绿也不用慌——每个 FAIL 都精确指向一个 bug，修完重跑。")
