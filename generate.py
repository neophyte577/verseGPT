import os
import torch
import argparse
import tiktoken
from model import GPT, GPTConfig

def load_tokenizer():
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    stoi = {i: i for i in range(vocab_size)}
    itos = {i: i for i in range(vocab_size)}
    return stoi, itos, enc

def generate_text(model, stoi, itos, enc, prompt="To be, or not to be", max_tokens=200, temperature=0.7, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(output_ids.squeeze().tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="out/verseGPT_best.pth", help="Path to trained model checkpoint")
    parser.add_argument("--prompt", type=str, default="To be, or not to be", help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering")
    args = parser.parse_args()
    
    stoi, itos, enc = load_tokenizer()
    vocab_size = len(stoi)
    
    config = GPTConfig(vocab_size=vocab_size, block_size=256, n_embd=256, n_layer=4, n_head=4, dropout=0.1)
    model = GPT(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))
    
    generated_text = generate_text(model, stoi, itos, enc, args.prompt, args.max_tokens, args.temperature, args.top_k)
    print("\nGenerated Text:\n")
    print(generated_text)

if __name__ == "__main__":
    main()
