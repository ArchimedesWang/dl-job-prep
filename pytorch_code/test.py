import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_qk, dim_v):
        super().__init__()
        self.q = nn.Linear(input_dim, dim_qk)
        self.k = nn.Linear(input_dim, dim_qk)
        self.v = nn.Linear(input_dim, dim_v)

        self.scale = math.sqrt(dim_qk)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.bmm(Q, K.transpose(1, 2))/self.scale

        weights = torch.softmax(scores, dim=-1)

        out = torch.bmm(weights, V)

        return out

if __name__ == "__main__":
    input_dim = 128
    dim_qk = 64
    dim_v = 64
    batch_size = 8
    seq_len = 10

    x = torch.randn(batch_size, seq_len, input_dim)
    attn = SelfAttention(input_dim = input_dim, dim_qk = dim_qk, dim_v = dim_v)
    out = attn(x)
    print(out.shape)
