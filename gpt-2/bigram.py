import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参数
batch_size = 32 # 每个批次的样本数量
block_size = 8 # 每个输入序列的长度
max_iters = 3000 # 最大迭代次数
eval_interval = 300 # 评估间隔
lr = 1e-2 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备选择，优先使用GPU
eval_iters = 200 # 评估迭代次数


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
    model.eval() # 设置模型为评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # 初始化损失张量
        for k in range(eval_iters):
            X, Y = get_batch(split) # 获取一个批次的数据
            logits, loss = model(X, Y) # 前向传播，计算输出和损失
            losses[k] = loss.item() # 记录损失
        out[split] = losses.mean() # 计算平均损失
    model.train() # 设置模型为训练模式
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # 获取嵌入向量
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
            logits, loss = self(idx) # 前向传播，获取输出
            logits = logits[:, -1, :] # 取最后一个时间步的logits
            probs = F.softmax(logits, dim=-1) # 计算概率分布
            idx_next = torch.multinomial(probs, num_samples=1) # 从概率分布中采样下一个索引
            idx = torch.cat((idx, idx_next), dim=1) # 将新索引添加到序列中
        return idx

model = BigramLanguageModel(vocab_size)
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