import torch
import torch.nn as nn
from torch.nn import functional as F
import math

max_iters = 3000
eval_interval = 300
eval_iters = 100
batch_size = 128
block_size = 128
n_embed = 192
n_heads = 6
lr = 1e-3
min_lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.2
n_layers = 4
warmup_iters = 100
lr_decay_iters = max_iters
# ---------------------


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() # 读取文本文件内容

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 為什么torch.stack对内部参数的要求是什么?
    # torch.stack要求输入的所有张量具有相同的形状和数据类型。它会在新的维度上将这些张量堆叠起来，从而形成一个新的张量。
    # 为什么data[i:i+block_size] for i in ix外面要加[]?
    # 因为torch.stack要求输入的所有张量具有相同的形状和数据类型。
    # 这里需要将一个列表中的张量堆叠成一个张量，所以需要加[]来创建一个列表。
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



def get_lr(iters):
    if iters < warmup_iters:
        return lr * (iters + 1) / (warmup_iters + 1)
    if iters > lr_decay_iters:
        return min_lr
    decay_ratio = (iters - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

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

class Head(nn.Module):
    def __init__(self, head_size, rope):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.rope = rope
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape   # B: batch_size, T: block_size, C: n_embed
        k = self.rope(self.key(x))
        q = self.rope(self.query(x))
        wei = q @ k.transpose(-2, -1) * self.head_size **-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, rope):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, rope) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out



class Block(nn.Module):
    def __init__(self, n_embed, n_heads, rope):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, rope)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = RMSNorm(n_embed)
        self.ln2 = RMSNorm(n_embed)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.rope = RotaryEmbedding(theta=10000, head_size=n_embed//n_heads, max_seq_len=block_size)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=n_heads, rope=self.rope) for _ in range(n_layers)])
        self.ln_f = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        x = self.blocks(token_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx
    

model = GPTLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-1, betas=(0.9, 0.95))

for iter in range(max_iters):
    # 1) 评估
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 2) 只设置学习率（这里才是 param_group 循环该待的地方）
    cur_lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    # 3) 前向
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    # 4) 反向 + 更新（三段式，必须在循环外）
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪
    optimizer.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
        