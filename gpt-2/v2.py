import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参数
batch_size = 512 # 每个批次的样本数量
block_size = 32 # 每个输入序列的长度
max_iters = 3000 # 最大迭代次数
eval_interval = 300 # 评估间隔
lr = 3e-4 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备选择，优先使用GPU
eval_iters = 200 # 评估迭代次数
n_embed = 48 # 嵌入向量的维度
n_heads = 4 # 注意力头的数量
n_layers = 4 # Transformer块的数量
dropout = 0.2 # dropout的概率
# --------------------------

torch.manual_seed(1337) # 设置随机种子，确保结果可复现

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() # 读取文本文件内容 

# 统计字符频率
chars = sorted(list(set(text))) # 获取文本中所有不同的字符，并排序
vocab_size = len(chars) # 计算不同字符的数量

# 输出所有不同的字符和数量
stoi = { ch:i for i,ch in enumerate(chars) } # 字符到索引的映射
itos = { i:ch for i,ch in enumerate(chars) } # 索引到字符的映射
encode = lambda s: [stoi[c] for c in s] # 将字符串编码为索引列表
decode = lambda l: ''.join([itos[i] for i in l]) # 将索引列表解码为字符串

# 将文本编码为整数索引，并转换为PyTorch张量,划分训练集和验证集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 计算训练集的长度
train_data = data[:n] # 训练集
val_data = data[n:] # 验证集

# 数据加载器
def get_batch(split):
    data = train_data if split == 'train' else val_data # 根据split选择数据集
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机选择起始索引
    x = torch.stack([data[i:i+block_size] for i in ix]) # 构建输入序列
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # 构建目标序列
    x, y = x.to(device), y.to(device) # 将数据移动到指定设备
    return x, y


# 装饰器是函数，它可以修改其他函数的行为。
# 装饰器是一个函数，它接受一个函数作为参数，并返回一个新的函数。

# @torch.no_grad()是在函数的开头添加了@torch.no_grad()装饰器，
# 没有它，函数内部的代码会计算梯度，而有了它，函数内部的代码不会计算梯度。

# 為什么会自动计算梯度，函数里没有写loss.backward()?
# 函数里调用了model(X, Y)，而model是一个nn.Module，它的forward方法会返回logits和loss，
# loss是一个张量，它的requires_grad属性默认为True，所以在计算loss时会自动计算梯度。

# 为什么要禁用梯度计算？
# 在评估模型性能时，我们不需要计算梯度，因为我们不会进行反向传播和参数更新。
# 禁用梯度计算可以节省内存和计算资源，提高评估速度。
@torch.no_grad() # 禁用梯度计算
def estimate_loss():
    out = {}
    # model.eval()是将模型设置为评估模式，这会影响某些层的行为，比如dropout和batch normalization。
    # 具体来说，dropout层在训练模式下会随机丢弃一些神经元，而在评估模式下会使用所有神经元。
    # batch normalization层在训练模式下会使用当前批次的均值和方差，而在评估模式下会使用整个训练集的均值和方差。
    model.eval() # 设置模型为评估模式
    for split in ['train', 'val']:
        # 为什么这里要用torch.zeros(eval_iters)来初始化损失张量？
        # 因为我们要在eval_iters次迭代中计算损失，并取平均值。
        losses = torch.zeros(eval_iters) # 初始化损失张量
        for k in range(eval_iters):
            X, Y = get_batch(split) # 获取一个批次的数据
            logits, loss = model(X, Y) # 前向传播，计算输出和损失
            losses[k] = loss.item() # 记录损失
        out[split] = losses.mean() # 计算平均损失
    model.train() # 设置模型为训练模式
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) # 线性层，用于生成键向量
        self.query = nn.Linear(n_embed, head_size, bias=False) # 线性层，用于生成查询向量
        self.value = nn.Linear(n_embed, head_size, bias=False) # 线性层，用于生成值向量
        # register_buffer是nn.Module的一个方法，用于注册一个持久缓冲区，该缓冲区不会被视为模型的参数，也不会在训练过程中更新。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 创建掩码张量
        self.dropout = nn.Dropout(dropout) # dropout层，防止过拟合

    def forward(self, x):
        B, T, C = x.shape # 获取批次大小、序列长度和嵌入维度
        k = self.key(x)   # (B, T, head_size) 生成键向量
        q = self.query(x) # (B, T, head_size) 生成查询向量
        # 为什么用C**-0.5来缩放注意力权重，而不是用head_size？
        # 因为C是嵌入维度，而head_size是每个注意力头的维度。缩放因子应该与查询和键的维度相关，而不是与嵌入维度相关。
        wei = q @ k.transpose(-2, -1) * self.head_size **-0.5 # (B, T, T) 计算注意力权重
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 应用掩码，防止未来信息泄露
        wei = F.softmax(wei, dim=-1) # (B, T, T) 归一化权重
        wei = self.dropout(wei) # 应用dropout
        v = self.value(x) # (B, T, head_size) 生成值向量
        out = wei @ v # (B, T, head_size) 计算加权和
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        # nn.ModuleList是一个有序的子模块容器，它可以像Python的list一样存储子模块，并且可以自动注册这些子模块。
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        # 为什么要把多头注意力的输出映射回嵌入维度?
        # 因为多头注意力是并行的，所以它的输出维度是n_heads * head_size，
        # 而嵌入维度是n_embed，所以需要把多头注意力的输出映射回嵌入维度。
        self.proj = nn.Linear(n_heads * head_size, n_embed) # 线性层，将多头注意力的输出映射回嵌入维度
        self.dropout = nn.Dropout(dropout) # dropout层，防止过拟合
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # 将每个头的输出拼接在一起
        out = self.dropout(self.proj(out)) # 应用投影层和dropout
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 线性层，将嵌入维度扩展为4倍
            nn.ReLU(), # 激活函数
            nn.Linear(4 * n_embed, n_embed), # 线性层，将维度缩小回嵌入维度
            nn.Dropout(dropout), # dropout层，防止过拟合
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
        def __init__(self, n_embed, n_heads):
            super().__init__()
            head_size = n_embed // n_heads # 每个注意力头的维度
            self.sa = MultiHeadAttention(n_heads, head_size) # 注意力头
            self.ffwd = FeedForward(n_embed) # 前馈神经网络
            self.ln1 = nn.LayerNorm(n_embed) # 层归一化
            self.ln2 = nn.LayerNorm(n_embed) # 层归一化

        def forward(self, x):
            x = self.sa(self.ln1(x)) + x # 残差连接，将注意力输出与输入相加
            x = self.ffwd(self.ln2(x)) + x # 残差连接，将前馈网络输出与输入相加
            return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # 位置嵌入表
        # 这行代码的*号是解包符，它会将列表中的元素解包，并将它们作为参数传递给函数。
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=n_heads) for _ in range(n_layers)]) # 堆叠多个Block
        self.ln_f = nn.LayerNorm(n_embed) # 层归一化
        self.lm_head = nn.Linear(n_embed, vocab_size) # 线性层，将嵌入向量映射到词汇表大小的输出
        self.lm_head.weight = self.token_embedding_table.weight # 将lm_head的权重与token_embedding_table共享
        
    def forward(self, idx, targets=None):
        B, T = idx.shape # 获取批次大小和序列长度

        
        token_emb = self.token_embedding_table(idx) # 获取嵌入向量, token_emb的形状为 (B, T, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # pos_emb的形状为 (T, n_embed)
        x = pos_emb + token_emb # 将位置嵌入和token嵌入相加, x的形状为 (B, T, n_embed)
        x = self.blocks(x) # 将嵌入向量输入到Transformer块中, x的形状为 (B, T, n_embed)
        x = self.ln_f(x) # 应用层归一化
        logits = self.lm_head(x) # logits的形状为 (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # 获取批次大小、序列长度和类别数
            logits = logits.view(B*T, C) # 调整形状以计算损失
            targets = targets.view(B*T) # 调整目标形状
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失
        return logits, loss

    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 为什么要有idx_cond = idx[:, -block_size:]这一行代码?
            # 因为idx的形状是(1, 1), 而block_size是8, 所以idx_cond的形状是(1, 8)
            # 为什么要取最后block_size个时间步的索引作为条件?
            # 因为我们需要预测下一个时间步的索引，所以需要用到之前的block_size个时间步的索引作为条件。
            # 这行代码和位置编码有什么关系?
            # 位置编码是用来表示时间步的信息，所以这里取最后block_size个时间步的索引作为条件是为了表示时间步的信息。
            idx_cond = idx[:, -block_size:] 
            # self(idx_cond)是调用BigramLanguageModel的forward方法，返回logits和loss。
            logits, loss = self(idx_cond) # 前向传播，获取输出
            logits = logits[:, -1, :] # 取最后一个时间步的logits
            probs = F.softmax(logits, dim=-1) # 计算概率分布
            idx_next = torch.multinomial(probs, num_samples=1) # 从概率分布中采样下一个索引
            idx = torch.cat((idx, idx_next), dim=1) # 将新索引添加到序列中
        return idx

model = BigramLanguageModel()
m = model.to(device) # 将模型移动到指定设备

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # 使用AdamW优化器

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss() # 评估损失
        # 输出训练和验证损失
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") 

    xb, yb = get_batch('train') # 获取一个训练批次
    logits, loss = model(xb, yb) # 前向传播，计算输出
    optimizer.zero_grad(set_to_none=True) # 清空梯度
    loss.backward() # 反向传播，计算梯度
    optimizer.step() # 更新参数

# 生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 初始化上下文
print(decode(m.generate(idx=context, max_new_tokens=500)[0].tolist())) # 生成并输出文本