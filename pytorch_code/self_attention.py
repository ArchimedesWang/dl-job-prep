import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_qk, dim_v):
        super().__init__()
        self.q = nn.Linear(input_dim, dim_qk)
        self.k = nn.Linear(input_dim, dim_qk)
        self.v = nn.Linear(input_dim, dim_v)
        
        # 缩放因子：通常是 dim_qk 的平方根
        self.scale = math.sqrt(dim_qk)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. 线性变换得到 Q, K, V
        Q = self.q(x) # (B, L, dim_qk)
        K = self.k(x) # (B, L, dim_qk)
        V = self.v(x) # (B, L, dim_v)

        # 2. 计算注意力得分 (Scores)
        # K.transpose(1, 2) 交换最后两个维度用于矩阵乘法
        # (B, L, dim_qk) * (B, dim_qk, L) -> (B, L, L)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        # 3. 归一化得到权重 (Weights)
        # dim=-1 表示在每一行（每个 query 对所有 key 的得分）上做 softmax
        weights = torch.softmax(scores, dim=-1)
        
        # 4. 加权求和得到输出 (Output)
        # (B, L, L) * (B, L, dim_v) -> (B, L, dim_v)
        out = torch.bmm(weights, V)
        
        return out

# --- 测试代码 ---
input_dim = 128
dim_qk = 64
dim_v = 64
batch_size = 2
seq_len = 10

# 正确实例化
attn = SelfAttention(input_dim, dim_qk, dim_v)
x = torch.randn(batch_size, seq_len, input_dim)
out = attn(x)

print("输入形状: ", x.shape)  # [2, 10, 128]
print("输出形状: ", out.shape) # [2, 10, 64]