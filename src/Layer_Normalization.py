"""Now let's implement layer normalization"""

class LayerNorm(nn.Module):
    def __init__(self, d_in, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_in))   # scale
        self.beta  = nn.Parameter(torch.zeros(d_in))  # shift
        self.eps = eps

    def forward(self, x):
        # x shape: (batch, context_length, d_in)

        mean = x.mean(dim=-1, keepdim=True)            # per token mean
        var  = x.var(dim=-1, keepdim=True, unbiased=False)  # per token variance

        x_norm = (x - mean) / torch.sqrt(var + self.eps)    # normalize
        return self.gamma * x_norm + self.beta               # scale + shift