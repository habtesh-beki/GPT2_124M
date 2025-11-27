import torch
from src.GPT_Tokenizer import GPT_Tokenizer
from src.Implement_DataLoader import Implement_DataLoader
from src.DataLoader import create_dataloaders
from src.GPT_model import GPTModel, GPT_CONFIG_124M
from src.Text_Generation import generate_text_simple
from src.Tokenizer import Tokenizer
from loss_calcuate_and_Entire_traning import calc_loss_loader 

"""calcuate cross-entropy loss"""

model = GPTModel(GPT_CONFIG_124M)

with open("iLoveMerge.txt", "r") as f :
   content = f.read()

print(content[:100])

print(len(content))

tokenizer = GPT_Tokenizer()
tokenized = tokenizer.encode(content)
print(len(tokenized))

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(content))
train_data = content[:split_idx]
val_data = content[split_idx:]


torch.manual_seed(123)

train_loader = Implement_DataLoader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = Implement_DataLoader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

print(len(train_loader))
print(len(val_loader))

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

# def calc_batch_loss(input_txt, target_txt, model, device):
#   input_txt , target_txt = input_txt.to(device), target_txt.to(device)
#   logits = model(input_txt)
#   loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
#   return loss
def calc_batch_loss(input_txt, target_txt, model, device):
    input_txt, target_txt = input_txt.to(device), target_txt.to(device)

    logits = model(input_txt)                    # (B, T, vocab)

    # Flatten both consistently
    logits = logits.flatten(0, 1)                # (B*T, vocab)
    target_txt = target_txt.flatten()            # (B*T)

    loss = torch.nn.functional.cross_entropy(logits, target_txt)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")


model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

"""### Traning Loop"""

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()

    context_size = model.pos_emb.weight.shape[0]

    encoded = torch.tensor(
        tokenizer.encode(start_context),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    decoded_text = tokenizer.decode(token_ids[0].tolist())
    print(decoded_text.replace("\n", " "))

    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    # Tracking history
    train_losses = []
    val_losses = []
    token_progress = []

    tokens_processed = 0
    step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()   # Enable training mode

        for inputs, targets in train_loader:
            # Reset previous gradients
            optimizer.zero_grad()

            # Forward + Loss
            loss = calc_batch_loss(inputs, targets, model, device)

            # Backpropagation
            loss.backward()

            # Parameter update
            optimizer.step()

            # Tracking progress
            tokens_processed += inputs.numel()
            step += 1

            # Run evaluation periodically
            if step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                token_progress.append(tokens_processed)

                print(
                    f"Epoch {epoch} | Step {step:06d} | "
                    f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}"
                )

        # Generate example output after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, token_progress

import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="It is never too late to", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")