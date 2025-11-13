"""### Implementing Embeddings
### Combines token embeddings with positional encodings to give the model
### information about both word identity and sequence position.
"""

import torch.nn as nn

class Embedding(nn.Module):
      def __init__(self, emb_dim, vocab_size, max_length):
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_length, emb_dim)

      def forward(self, x):
          positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
          return self.token_emb(x) + self.pos_emb(positions)