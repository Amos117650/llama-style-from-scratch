# gqa_selftest.py —— GQA 五关考卷（裁判版）
#
# 用法：在本文件所在目录运行
#   /e/anaconda3/envs/cs336/python.exe gqa_selftest.py
#
# 设计说明：
# 1. 不整模块 import testv5（训练循环在模块顶层，import 即开训）；
#    只 exec 配置区 + RMSNorm/RotaryEmbedding/MultiHeadAttention 三段源码。
# 2. 裁判 referee_forward 是独立实现：逐头 Python 循环 + 逐头 3 维 rope，
#    与被测的向量化路径无任何共享代码。两边 allclose = 交叉验证成立。
# 3. M2 未实现（MultiHeadAttention 无 n_kv_heads 参数）时自动只跑关 1~3。
# 4. 考卷也是代码、会出错（Bug 本第 2 条）。挂了先怀疑考卷，再用独立写法裁决。
# 5. 若你在 M2 改了构造签名/属性名（如 q_proj/k_proj），改下方「适配区」即可。

import sys
import inspect
import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SRC = (HERE / 'testv5.py').read_text(encoding='utf-8')
ns = {}
exec(SRC[:SRC.index("with open(")], ns)  # imports + 超参数
start = SRC.index("def repeat_kv") if "def repeat_kv" in SRC else SRC.index("class RMSNorm")
exec(SRC[start:SRC.index("class FeedForward")], ns)  # repeat_kv + RMSNorm/RotaryEmbedding/MHA

MultiHeadAttention = ns['MultiHeadAttention']
RotaryEmbedding = ns['RotaryEmbedding']
N_EMBED, N_HEADS, BLOCK = ns['n_embed'], ns['n_heads'], ns['block_size']
HEAD = N_EMBED // N_HEADS

# ---------------- 适配区：若签名/属性名变了，只改这里 ----------------
ATTR_Q, ATTR_K, ATTR_V, ATTR_P = 'query', 'key', 'value', 'proj'
HAS_GQA = 'n_kv_heads' in inspect.signature(MultiHeadAttention.__init__).parameters

def make_rope():
    return RotaryEmbedding(theta=10000, head_size=HEAD, max_seq_len=BLOCK)

def build(n_kv_heads=None):
    rope = make_rope()
    if n_kv_heads is None or not HAS_GQA:
        return MultiHeadAttention(N_HEADS, N_EMBED, rope)
    try:
        return MultiHeadAttention(N_HEADS, N_EMBED, rope, n_kv_heads)
    except TypeError:
        return MultiHeadAttention(N_HEADS, N_EMBED, rope, n_kv_heads=n_kv_heads)
# --------------------------------------------------------------------

def referee_forward(x, qw, kw, vw, pw, pb, rope, n_kv):
    """独立裁判：逐头循环、逐头 3 维 rope。n_kv = H 时即 MHA。"""
    B, T, C = x.shape
    H = qw.shape[0] // HEAD
    G = H // n_kv
    tril = torch.tril(torch.ones(T, T))
    outs = []
    for i in range(H):
        kv = i // G  # 分组映射：第 i 个 q 头吃第 i//G 个 kv 头
        qh = rope(x @ qw[i * HEAD:(i + 1) * HEAD].T)       # (B,T,head) 过 rope
        kh = rope(x @ kw[kv * HEAD:(kv + 1) * HEAD].T)
        vh = x @ vw[kv * HEAD:(kv + 1) * HEAD].T            # v 不过 rope
        wei = qh @ kh.transpose(-2, -1) * HEAD ** -0.5
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        outs.append(wei @ vh)
    out = torch.cat(outs, dim=-1) @ pw.T
    return out + pb if pb is not None else out

def weight_of(m, name):
    lin = getattr(m, name)
    return (lin.weight.data, lin.bias.data if lin.bias is not None else None)

results = []

def check(name, ok, detail=""):
    results.append(ok)
    print(("✅" if ok else "❌") + f" {name}" + (f"  [{detail}]" if detail else ""))

# ================= 关 1：参数对账 =================
def guan1(expect_kv):
    m = build(expect_kv if HAS_GQA else None)
    got = sum(p.numel() for p in m.parameters())
    # proj 沿用旧代码默认 bias=True（+d）；q/k/v 为 bias=False
    want = (N_EMBED * N_EMBED * 2                       # q + proj 权重
            + 2 * N_EMBED * expect_kv * HEAD            # k + v 权重
            + N_EMBED)                                  # proj bias
    check(f"参数对账 (n_kv={expect_kv})", got == want,
          f"实际 {got:,} / 期望 {want:,} = d²·2 + 2·d·{expect_kv}·head + d(bias)")
    return m

# ================= 关 2：RoPE 压平一致性 =================
def guan2():
    torch.manual_seed(3407)
    rope = make_rope()
    B, T = 2, 64
    x = torch.randn(B, N_HEADS, T, HEAD)
    path_a = rope(x.reshape(-1, T, HEAD)).reshape(B, N_HEADS, T, HEAD)   # 压平喂入
    path_b = torch.stack([rope(x[:, h]) for h in range(N_HEADS)], dim=1) # 逐头喂入
    check("RoPE 压平一致 (B·H,T,D) ≡ 逐头 (B,T,D)",
          torch.allclose(path_a, path_b, atol=1e-6),
          f"max|Δ|={(path_a - path_b).abs().max():.2e}")

# ================= 关 3：退化等价 K=N ≡ MHA =================
def guan3():
    torch.manual_seed(3407)
    m = build(N_HEADS if HAS_GQA else None).eval()
    x = torch.randn(2, 48, N_EMBED)
    with torch.no_grad():
        mine = m(x)
        qw, kw, vw = (weight_of(m, a)[0] for a in (ATTR_Q, ATTR_K, ATTR_V))
        pw, pb = weight_of(m, ATTR_P)
        ref = referee_forward(x, qw, kw, vw, pw, pb, m.rope, n_kv=N_HEADS)
    check("退化等价：K=N 时与独立 MHA 裁判逐元素一致",
          torch.allclose(mine, ref, atol=1e-5),
          f"max|Δ|={(mine - ref).abs().max():.2e}")

# ================= 关 4：GQA 形状 + 整除断言（M2） =================
def guan4():
    # 默认构造：模型接线（Block 不传 n_kv_heads）走的就是这条路，必须回落 MHA 不崩
    try:
        m0 = MultiHeadAttention(N_HEADS, N_EMBED, make_rope()).eval()
        x0 = torch.randn(2, 48, N_EMBED)
        with torch.no_grad():
            out0 = m0(x0)
        check("默认构造（n_kv_heads 不传）回落 MHA 不崩",
              out0.shape == (2, 48, N_EMBED),
              "排查点：assert 是否误用了原始参数 n_kv_heads（None）")
    except TypeError as e:
        check("默认构造（n_kv_heads 不传）回落 MHA 不崩", False, f"TypeError: {e}")

    m = build(n_kv_heads=2).eval()
    x = torch.randn(2, 48, N_EMBED)
    with torch.no_grad():
        out = m(x)
    ok_shape = out.shape == (2, 48, N_EMBED)
    check("GQA 前向形状 (B,T,d) 不变", ok_shape, f"{tuple(out.shape)}")
    kw = weight_of(m, ATTR_K)[0]
    check("k 投影输出 = n_kv_heads·head_size", kw.shape == (2 * HEAD, N_EMBED),
          f"{tuple(kw.shape)}，期望 ({2 * HEAD},{N_EMBED})")
    try:
        build(n_kv_heads=4)  # 6 % 4 != 0，应被断言拒绝
        check("整除断言 n_heads % n_kv_heads == 0", False, "6/4 未被拒绝——缺断言")
    except (AssertionError, ValueError, RuntimeError):
        check("整除断言 n_heads % n_kv_heads == 0", True, "6/4 正确被拒")

# ================= 关 5：分组映射 i//G + K=1 退化（M2） =================
def guan5():
    for n_kv in (2, 1):
        torch.manual_seed(3407)
        m = build(n_kv_heads=n_kv).eval()
        x = torch.randn(2, 48, N_EMBED)
        with torch.no_grad():
            mine = m(x)
            qw, kw, vw = (weight_of(m, a)[0] for a in (ATTR_Q, ATTR_K, ATTR_V))
            pw, pb = weight_of(m, ATTR_P)
            ref = referee_forward(x, qw, kw, vw, pw, pb, m.rope, n_kv=n_kv)
        tag = f"K={n_kv}（{'GQA 6:2' if n_kv == 2 else 'MQA 退化'}）"
        check(f"分组映射 {tag} 与裁判一致（抓 i//G 写成 i%G）",
              torch.allclose(mine, ref, atol=1e-5),
              f"max|Δ|={(mine - ref).abs().max():.2e}")

# ================= 主流程 =================
if __name__ == '__main__':
    print(f"被测：testv5.MultiHeadAttention  d={N_EMBED} H={N_HEADS} head={HEAD} block={BLOCK}")
    print(f"模式：{'M1+M2 全五关' if HAS_GQA else 'M1 三关（未检出 n_kv_heads，关 4/5 跳过）'}\n")

    guan1(N_HEADS)
    guan2()
    guan3()
    if HAS_GQA:
        guan1(2)
        guan4()
        guan5()

    passed, total = sum(results), len(results)
    print(f"\n==== 考卷结果：{passed}/{total} ====")
    sys.exit(0 if passed == total else 1)
