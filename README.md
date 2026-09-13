# llama-style-from-scratch

从 GPT-2 风格出发，逐组件手写迁移到 LLaMA / Qwen 风格 Transformer 的学习仓库。

每个组件走同一套流程：**原理 → 手写 → 对比验证 → 讲清楚**；全部 A/B 实验在同一个迷你实验台（Tiny Shakespeare）上完成，组件正确性由独立的性质测试考卷验收，不靠训练曲线「看起来收敛」。

## 迷你实验台

| 项 | 值 |
|---|---|
| 数据 | Tiny Shakespeare（字符级，vocab=65） |
| d_model | 192 |
| 层数 | 4 |
| 注意力头 | 6 query / 2 KV（GQA，可切 MHA） |
| 上下文长度 | 128 |
| 训练 | 3,000 iters，AdamW（cosine + warmup），CPU/单卡可跑 |

## 架构演进：GPT-2 风格 → LLaMA / Qwen 风格

| 组件 | GPT-2 风格 | 本仓库 | 验证方式 |
|---|---|---|---|
| 位置编码 | 可学习绝对位置嵌入 | **RoPE**：半分式配对（NeoX/HF 式）、只作用于 q/k、θ=10000、fp32 现算 | 性质三关：保范数 / 平移不变 / Toeplitz |
| 归一化 | LayerNorm | **RMSNorm**（可学习缩放 `(1+w)`，fp32 计算） | 与绝对位置版 val loss ±5% 带内 |
| FFN | GELU，4·d² | **SwiGLU**，hidden = ⌊8/3·d⌋ = 512 | 参数对账 3·d·hidden = 8·d² 严格相等；val −2.6% |
| 注意力 | MHA 6 头 | **GQA 6:2**（`repeat_kv`：expand→reshape，`i//group` 连续块分组） | 分组映射 max\|Δ\|=0；val Δ=0.0%；KV cache 3× |

## 实测数字（val loss @3,000 iters）

单次运行、无固定种子，作量级参考；各检查点方向一致。

| 对照 | A | B | 结论 |
|---|---|---|---|
| FFN | GELU 1.5983 | SwiGLU 1.5572 | **−2.6%** |
| KV 头数 | MHA 6:6 = 1.5752 | GQA 6:2 = 1.5752 | **Δ=0.0%**；参数 1,784,513 → 1,587,905（−196,608）；KV cache 6,144 → 2,048 B/token（fp32，**3×**） |

> GQA 论文的核心主张在本仓库小尺度上可复现：质量基本不损，推理侧 KV cache 线性下降。省下的参数预算真实 LLM 通常再分配给宽度/深度，所以「GQA 几乎不减参数」的说法只在孪生对照口径下成立。

## 与 Qwen2 / Qwen2.5-VL 源码的对照

逐行阅读 transformers 官方实现（`modeling_qwen2.py` / `modeling_qwen2_5_vl.py`）后整理的差异精选：

- **投影 bias**：Qwen2 解码器 = q/k/v_proj 有 bias + o_proj 无；视觉塔为 fused qkv 单 Linear 但等头数 MHA；
- **RoPE**：θ 由 config 决定（Qwen2=10000，Qwen2.5 系=1e6）；每步 forward 现算 + 全宽 `cat(freqs, freqs)`；配对为 rotate_half 半分式，与 RoFormer 原文交错式**不可混用**；
- **mask**：加性 0/-inf 动态构造（vs `tril` + `masked_fill`）；**视觉塔 `is_causal=False`——图像内双向注意力**；
- **M-RoPE**（Qwen2.5-VL 文本侧）：token 位置为 (t,h,w) 三元组，`mrope_section=[16,24,24]` 三段连续区段各配一轴；纯文本退化为 (m,m,m)。

## LoRA（进行中）

`lora.py`：loralib 风格 `Linear(nn.Linear, LoRALayer)`——A:(r,in) kaiming 初始化 / B:(out,r) 全零（起点等价）、scaling=α/r、构造即冻结 W₀、`train()/eval()` 自动 merge/unmerge（幂等，`merged` 标志防重复加减）。

验收五关：形状与初始化 / **B=0 起点等价**（输出与裸 Linear 逐位相等）/ scale 数学（含 scale=1 反事实）/ **merge 双向等价**（合并路径 vs 旁路路径 <1e-5，实测 1.3e-6）/ 参数对账与梯度路由（冻结件零梯度、第一步只有 B 动）。

## 验证方法：每个组件三关

1. **数学关**——独立性质测试：考卷与实现分离、各写各的算法路径（考卷本身也是代码、也出过错，靠交叉验证裁决）。性质测试考卷本地留存，未随库发布；
2. **经验关**——同一实验台 A/B 训练，val loss ±5% 容忍带判定；
3. **默写关**——白纸重建组件并运行（`rope.py` / `swiglu.py` / `gqa.py` 即默写件，其中 gqa.py 自带 `__main__` 自测；首跑抓出的真实 bug 是这套流程价值的最好证据）。

## 进行中

- LoRA 挂载训练（q/v 投影，r=8、α=16），目标 merge 前后 logits <1e-5 + 收敛对照
- KV cache 解码路径
- RoPE 按 Qwen 三件套职责分解重构（`RotaryEmbedding` 只产 cos/sin + `rotate_half` / `apply_rotary_pos_emb(q,k,cos,sin)` 自由函数）

## 运行

```bash
pip install torch

# 数据：Tiny Shakespeare 放到 input.txt（已 gitignore）
curl -LO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python testv5.py          # 训练迷你模型 + 生成样本
python gqa.py             # 默写件自带 __main__ 性质自测（repeat_kv 玩具分组 + KV cache 字节对账）
```

## 归属说明

本仓库为学习仓库，遵循个人学习协议：AI 助手承担审查、出考卷、跑验证的裁判角色，组件的实现理解与白纸复现由本人完成。

| 文件 | 归属 |
|---|---|
| testv3.py / testv4.py / testv5.py | 本人手写（AI 审查 + 考卷验收） |
| rope.py / swiglu.py / gqa.py | 本人白纸默写件（rope.py 后对齐 HF 源码约定重构） |
| lora.py / testv6.py | AI 辅助生成参考实现，本人手动录入并逐行注释理解 |
| README.md | AI 助手（ZCode）起草 |

验证考卷与学习笔记不入库（本地保存）；考卷由 AI 助手编写，用于组件验收。

## License

MIT
