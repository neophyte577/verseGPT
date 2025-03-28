import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from contextlib import nullcontext
from model import GPT, GPTConfig
from lion_pytorch import Lion
import tiktoken

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()  

out_dir = 'out'
eval_interval = 1000
log_interval = 10
eval_only = False
always_save_checkpoint = True
gradient_accumulation_steps = 2
batch_size = 32
block_size = 256
learning_rate = 5e-5
max_iters = 10000
weight_decay = 1e-2
grad_clip = 1.0
early_stopping_patience = 5
n_embd = 256
n_layer = 4
n_head = 4

torch.manual_seed(1337)
if torch.cuda.is_available():
    cudnn.benchmark = True  

torch.set_num_threads(torch.get_num_threads() * 4)

def load_data():
    with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    tokenized_text = enc.encode(text)
    vocab_size = enc.n_vocab
    stoi = {i: i for i in range(vocab_size)}
    itos = {i: i for i in range(vocab_size)}
    data = torch.tensor(tokenized_text, dtype=torch.long)
    return data, stoi, itos, vocab_size

data, stoi, itos, vocab_size = load_data()

config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=0.1)
model = GPT(config).to(device)
model.apply(model._init_weights)
optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scaler = torch.amp.GradScaler("cuda") if use_amp else None

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device, non_blocking=True).long(), y.to(device, non_blocking=True).long()

def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def train():
    best_loss = float('inf')
    patience = 0
    try:
        for iter in range(max_iters):
            model.train()
            optimizer.zero_grad()

            # conditional AMP usage
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                for _ in range(gradient_accumulation_steps):
                    x, y = get_batch(data)
                    x, y = x.to(device), y.to(device)  
                    _, loss = model(x, y)
                    loss = loss / gradient_accumulation_steps
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # apply optimizer step conditionally based on AMP usage
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if iter % log_interval == 0:
                print(f"Iter {iter}: Train Loss {loss.item():.4f}")

            # periodic evaluation
            if iter % eval_interval == 0 and iter > 0:
                val_loss = evaluate()
                print(f"Iter {iter}: Val Loss {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    if always_save_checkpoint:
                        os.makedirs(out_dir, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(out_dir, 'verseGPT_best.pth'))
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        print("Early stopping triggered.")
                        break
                    
    except KeyboardInterrupt:
        print("Training interrupted. Saving latest model as best checkpoint...")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, 'verseGPT_best.pth'))
        print("Model saved as verseGPT_best.pth.")

if __name__ == "__main__":

    if os.path.exists('./out'):
        shutil.rmtree('./out')
        os.mkdir('out')

    train()
