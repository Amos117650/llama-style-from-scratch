import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from swiglu import SwiGLU
from rope import RotaryEmbedding
from gqa import repeat_kv
# ---------------------
max_iters = 3000
eval_interval = 300
eval_iters = 100
batch_size = 128
block_size = 128
n_embed = 192
n_heads = 6
n_kv_heads = 2
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



class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embed, rope, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.embed_size = n_embed
        self.head_size = n_embed // n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0
        self.group_size = n_heads // self.n_kv_heads
        self.query = nn.Linear(n_embed, self.n_heads*self.head_size, bias=False)
        self.key = nn.Linear(n_embed, self.n_kv_heads*self.head_size, bias=False)
        self.value = nn.Linear(n_embed, self.n_kv_heads*self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(n_heads * self.head_size, n_embed)
        self.rope = rope
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape   # B: batch_size, T: block_size, C: n_embed
        # flatten(0, 1)的作用是将第0维和第1维合并为一个维度，从而将形状从(B, n_heads, T, head_size)变为(B*n_heads, T, head_size)。
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2).flatten(0, 1) # (B*n_heads, T, head_size)
        k = self.key(x).view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2).flatten(0, 1) # (B*n_kv_heads, T, head_size)
        v = self.value(x).view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2) # (B, n_kv_heads, T, head_size)

        q = self.rope(q).view(B, self.n_heads, T, self.head_size)   # (B, n_heads, T, head_size)
        k = self.rope(k).view(B, self.n_kv_heads, T, self.head_size)   # (B, n_kv_heads, T, head_size)

        k = repeat_kv(k, self.group_size)   # (B, n_heads, T, head_size)
        assert k.shape == q.shape
        # 计算注意力权重
        wei = q @ k.transpose(-2, -1) * self.head_size **-0.5
        # 把self.tril[:T, :T] == 0广播到所有头
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 为什么能保证softmax是对每一行做归一化的?
        # 这是因为在计算注意力权重时，wei的形状是(B, n_heads, T, T)，其中最后一个维度表示每个查询位置对应的所有键位置的权重。
        # F.softmax函数默认会对指定维度（这里是最后一个维度）进行归一化，因此每一行（即每个查询位置对应的所有键位置的权重）都会被归一化，使得它们的和为1。
        # 就像是一个(T, T)的矩阵，dim=-1在执行softmax时会对每一行进行归一化。
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        v = repeat_kv(v, self.group_size)
        assert v.shape == q.shape
        out = wei @ v   # (B, n_heads, T, head_size)
        # .contiguous()的作用是确保张量在内存中是连续的，从而提高计算效率。
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_size)
        out = self.resid_dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            SwiGLU(n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, rope, n_kv_heads=None):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads, n_embed, n_kv_heads=n_kv_heads, rope=rope)
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
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=n_heads, rope=self.rope, n_kv_heads=n_kv_heads) for _ in range(n_layers)])
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
print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")

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
        