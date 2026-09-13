import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALayer():
    def __init__(self, r, lora_alpha, lora_dropout, merge_weights=True):
        self.r = r
        self.lora_alpha = lora_alpha

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merge_weights = merge_weights  # eval()时是否合并权重 
        self.merged = False # 是否已经合并权重

class Linear(nn.Linear, LoRALayer):
    """带 LoRA 旁路的 nn.Linear。

    未 merge 时前向：
        h = W x + (x @ Aᵀ @ Bᵀ) * scaling      # scaling = alpha / r
    W:(out,in) 冻结，A:(r,in)、B:(out,r) 可训。
    B 初始化为 0 ⇒ B·A = 0 ⇒ 出厂时与原 nn.Linear 输出完全一致（不破坏预训练）。
    """
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0.0, merge_weights=True, bias=True):
        # 为什么先调 nn.Linear.__init__ 再调 LoRALayer.__init__？
        # 前者建出 self.weight/self.bias，后者只塞普通字段；顺序不能反。
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)

        if r > 0:   # 如果 r > 0，才添加 LoRA 参数
            # 为什么用 self.weight.new_zeros(...) 而不是 torch.zeros(...)？
            # new_zeros 继承 weight 的 dtype 和 device——模型半精度/上 GPU 时 A、B 自动跟上。
            # torch.weight.new_zeros((r, in_features))为什么有两个括号？
            # 因为 new_zeros 的参数是一个元组，表示张量的形状。这里我们希望创建一个形状为 (r, in_features) 的张量，所以需要使用两个括号来传递一个元组。
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # 冻结预训练权重：这一层从此只训 A、B
            self.weight.requires_grad = False

        # 为什么末尾还要再调一次 reset_parameters？
        # nn.Linear.__init__ 内部已调过一次 self.reset_parameters()——那时走的是
        # 本类覆写版，但 lora_A 还没注册，hasattr 为 False，只初始化了 weight。
        # 现在补一次，让 lora_A/lora_B 的初始化也走同一个入口。
        self.reset_parameters()

    def reset_parameters(self):
        # 先调 nn.Linear.reset_parameters() 初始化 weight/bias
        nn.Linear.reset_parameters(self)
        # 再初始化 lora_A/lora_B
        # if hasattr(self, 'lora_A')是什么意思？
        # hasattr(self, 'lora_A') 检查当前实例是否有属性 'lora_A'。
        # 如果有，说明 LoRA 参数已经被创建，可以对它们进行初始化。
        if hasattr(self, 'lora_A'):
            # A 的初始化和 nn.Linear 对 weight 的默认初始化完全一致（kaiming_uniform, a=√5）
            # B 必须 0 初始化：保证 B@A = 0，起点不破坏预训练模型（论文的关键细节）
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    # 为什么要用property？
    # property 装饰器将方法变为属性访问，这样可以通过 obj.delta_weight 直接访问，而不需要调用方法 obj.delta_weight()。这使得代码更简洁和直观，尤其是在需要频繁访问该属性的情况下。
    @property   
    def delta_weight(self):
        """ΔW = B·A·scaling，形状 (out, in)，与 weight 同形状，可直接加减。"""
        return (self.lora_B @ self.lora_A) * self.scaling

    def train(self, mode=True):
        # 覆写 train()：train/eval 切换时自动 merge/unmerge——
        #   model.eval()  → W += ΔW 并进去，推理时一步 F.linear，零额外延迟
        #   model.train() → W -= ΔW 拆回来，训练时梯度走旁路
        # merged 标志保证幂等：反复调用不会重复加减。
        # 为什么要在 train 方法中调用 nn.Linear.train(self, mode)？
        # 调用 nn.Linear.train(self, mode) 是为了确保父类 nn.Linear 的 train 方法被正确调用。
        nn.Linear.train(self, mode)
        # 为什么 train 模式拆回旁路、eval 模式合并？
        # eval（推理）合并：前向只算一次 F.linear(x, W+ΔW)，零旁路开销；
        # train（训练）拆回：梯度必须走旁路才能更新 A、B——合并态的前向不经过 lora_A/lora_B，训了也白训。 
        if mode:
            if self.merge_weights and self.merged:
                self.weight.data -= self.delta_weight  # 先解开
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                self.weight.data += self.delta_weight  # 再合并
                self.merged = True

        return self # 返回当前实例，便于链式调用

    def forward(self, x):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, self.bias)    # # 主干：W x（W 已冻结）
            # 旁路：x @ Aᵀ @ Bᵀ * scaling，形状 (…, in) → (…, r) → (…, out)
            # 为什么 A、B 分开乘而不是先算 B@A？
            # 分开乘是 O(…·in·r + …·r·out)；先合成 ΔW 再乘是 O(…·in·out)——低秩就白搭了
            result = result + (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling 
            return result

        return F.linear(x, self.weight, self.bias)  # W x + b，推理时一步到位

def mark_only_lora_as_trainable(model, bias='none'):
    """冻结除 LoRA 外的所有参数，节省显存。
    bias 三档（官方语义）：
        'none'      —— bias 一并冻结（最常用）
        'all'       —— 所有 bias 解冻一起训
        'lora_only' —— 只解冻“挂了 LoRA 的那些层”的 bias
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    if bias  == 'none':
        return
    elif bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
    elif bias == 'lora_only':
        # model.modules() 会递归遍历所有子模块，返回每个子模块的实例
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError

def lora_state_dict(model, bias='none'):
    """只收集 LoRA 权重——checkpoint 体积从 O(全量参数) 降到 O(rank 参数)。"""
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return


