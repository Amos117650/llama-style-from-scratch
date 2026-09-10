def repeat_kv(x, group_size):
    B, n_kv_head, T, head_size = x.shape
    x = x.unsqueeze(2).expand(-1, -1, group_size, -1, -1)   # 
    x = x.reshape(B, n_kv_head * group_size, T, head_size)
    return x



def kv_cache_bytes(n_layers, n_kv_heads, head_size, seq_len):
    """每个 token 的 KV cache 字节数（fp32，4 B/元素）。

    每 token、每层、每个 kv 头存 K、V 两份，各 head_size 个元素：
        n_layers × n_kv_heads × 2 × head_size × 4
    seq_len 不参与——这是"每 token"口径；整个序列再乘 seq_len（batch 再乘 B）。
    
    2 × head_size：每个 kv 头存 K、V 两份，各 head_size 个元素
    × 4：fp32 每元素 4 字节（如果用 bf16 就是 ×2）
    seq_len 不乘：断言口径是“每 token”（4 × 6 × 2 × 32 × 4 = 6144）。
    要算整个序列的 cache 总量再乘 seq_len，有 batch 再乘 batch size。
    Pylance 那个“未存取 seq_len”的 Hint 就是这个原因，属于刻意保留——参数留着是为了和测试签名对齐
    GQA 缩减比直接来自公式对 n_kv_heads 线性：6 头 → 2 头正好 3×
    """
    return n_layers * 2 * n_kv_heads * head_size * 4


if __name__ == '__main__':
    # 原理：直接运行本文件时 __name__ == '__main__' 为真，自测执行；
    # 被导入时（from gqa import repeat_kv）__name__ == 'gqa'，自测跳过。
    # testv5.py 的训练循环 W2 前也要套这个壳。
    import torch

    # —— 自测 1：repeat_kv 分组语义（10/20 玩具检查）——
    k = torch.zeros(1, 2, 3, 4)   # B=1, n_kv=2, T=3, head=4
    k[0, 0] = 10.0                # kv 头 0 全填 10
    k[0, 1] = 20.0                # kv 头 1 全填 20
    out = repeat_kv(k, group_size=3)
    assert out.shape == (1, 6, 3, 4), f"形状错: {tuple(out.shape)}"
    heads = [out[0, i, 0, 0].item() for i in range(6)]
    assert heads == [10.0] * 3 + [20.0] * 3, f"分组顺序错: {heads}（i%G 型错误）"
    print("✅ repeat_kv：形状 + i//group 分组语义正确")

    # —— 自测 2：KV cache 对账（3 倍）——
    mha = kv_cache_bytes(n_layers=4, n_kv_heads=6, head_size=32, seq_len=128)
    gqa = kv_cache_bytes(n_layers=4, n_kv_heads=2, head_size=32, seq_len=128)
    assert mha == 6144, f"MHA 每 token 应 6,144 B，实际 {mha:,}"
    assert mha / gqa == 3, f"缩减比应为 3，实际 {mha / gqa}"
    print(f"✅ KV cache：MHA {mha:,} B/token → GQA {gqa:,} B/token（3×）")

    print("==== gqa.py 默写关自测全绿 ====")