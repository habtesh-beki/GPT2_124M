import tiktoken

class GPT_Tokenizer:
   def __init__(self):
      self.enc = tiktoken.get_encoding("gpt2")

   def encode(self, text):
       self.encoding = self.enc.encode(text)
       return self.encoding

   def decode(self, token):
       self.decoding = self.enc.decode(token)
       return self.decoding

"""GPT-2 tokenizer implementation using OpenAI's tiktoken library.
Provides encoding and decoding functionality compatible with OpenAI's GPT models.
"""

with open("iLoveMerge.txt", "r", encoding="utf-8") as file:
   raw_data = file.read()

"""Read the text file containing training data for tokenization and model training"""

text_data = raw_data[:100]
print(text_data)

"""### For demonstration, using the first 100 characters which will be encoded later"""

tokenizer = GPT_Tokenizer()
encode_text = tokenizer.encode(text_data)
print(encode_text)

"""### Tokenization complete: words converted to tokens using tiktoken"""

decode_text = tokenizer.decode(encode_text)
print(decode_text)

"""### Decoding reproduces the exact original text that was encoded

## Prepares the dataset for LLM training using PyTorch's DataLoader.

### This structures the data into batches, enabling efficient GPU utilization
### and shuffling. For next-token prediction (causal LM), the target sequence
### is the input sequence shifted right by one position.
"""
