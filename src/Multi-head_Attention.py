import torch
torch.manual_seed(123)
import torch.nn as nn
from src.GPT_Tokenizer import GPT_Tokenizer

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Ensure even split across heads
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Linear projections for Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final output projection
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular = 1 -> masked)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # Reshape for multi-head: (b, seq_len, num_heads, head_dim)
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (b, num_heads, seq_len, seq_len)
        scores = Q @ K.transpose(2, 3)

        # Causal masking
        mask = self.mask.bool()[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over last dimension
        attn = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        context = attn @ V   # (b, num_heads, seq_len, head_dim)

        # Merge heads back: (b, seq_len, d_out)
        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.d_out)

        return self.out_proj(context)
    



inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

batch_size, context_length, d_in = batch.shape
d_out = 6
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

text_demo = "Hello Every one"
tokenizer = GPT_Tokenizer()
tokenized = tokenizer.encode(text_demo)
print(tokenized)