import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 512
n_heads = 4
embed_size = 64
max_iters = 3000
eval_interval = 300
eval_iters = 200
block_size = 8
dropout = 0.2
n_layers = 4
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------

torch.manual_seed(1337) # 设置随机种子，确保结果可复现

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() # 读取文本文件内容

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1: i +block_size + 1] for i in ix])
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



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        # 为什么是block_size而不是embed_size?
        # 因为在计算注意力权重时，我们需要对每个位置的查询向量和所有位置的键向量进行点积操作，
        # 所以需要一个block_size * block_size的掩码矩阵来屏蔽未来的信息，防止模型在训练时看到未来的单词。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x)
        # 为什么先做点积再做掩码?
        # 因为我们需要先计算每个位置的查询向量和所有位置的键向量的相似度，然后再根据掩码矩阵屏蔽未来的信息。
        wei = q @ k.transpose(-2, -1) * self.head_size **-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(n_heads))
        self.proj = nn.Linear(n_heads * head_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Feedforward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

    
class Block(nn.Module):
    def __init__(self, embed_size, n_heads):
        super().__init__()
        self.head_size = embed_size // n_heads
        self.sa = MultiHeadAttention(n_heads, self.head_size)
        self.ffwd = Feedforward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = self.ln1(self.sa(x)) + x
        x = self.ln2(self.ffwd(x)) + x
        return x

class RMSNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_size))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, n_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = RMSNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight # 共享权重，将线性层的权重与嵌入表的权重共享

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + position_emb
        x = self.blocks(x)
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
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # idx的形状是(1, 1) -> (1, 2) -> (1, 3) -> ... -> (1, 501)
        return idx
           


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") 

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
# 为什么会有[0].tolist()这一行代码?
# 因为model.generate(context, max_new_tokens=500)返回的是一个张量，形状为(1, 501)，
# 而我们只需要取第一个元素，也就是生成的文本的索引序列，所以用[0]取第一个元素，然后用tolist()将其转换为列表，方便后续的解码操作。
# 是不是说[0]就是把第0行所有元素取出来，tolist()就是把张量转换为列表?
# 是的，[0]是取张量的第0行，tolist()是将张量转换为Python列表，这样可以方便地进行后续处理，比如解码成文本。
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    

