# 🎭**verseGPT: A Lightweight Shakespearean Text Generator**

**verseGPT** is a small-scale, GPT-style language model trained on a tiny dataset of Shakespearean verse. The goal of this project is to generate convincing Elizabethan verse in response to English-language prompts using a lightweight model architecture trained on a minimal dataset.

## **Project Overview**
This project was inspired by the challenge of producing high-quality, stylistically constrained text generation with a small-scale GPT model trained on extremely limited data. Unlike large language models trained on massive datasets, verseGPT aims to:
- Be efficient and lightweight, making it practical for small-scale experimentation
- Use causal self-attention to generate text autonomously, without peeking at future tokens
- Be trained exclusively on tiny_shakespeare.txt, a .txt corpus consisting of Shakespeare’s collected works

## **Features**
- **Custom GPT Model**: Based heavily on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), with modifications for even smaller-scale training
- **Causal Self-Attention**: Ensures autoregressive text generation in Shakespearean style
- **Efficient Training Pipeline**: Supports CPU & GPU training with automatic mixed precision (AMP) for faster computation
- **Configurable Hyperparameters**: Easily modify model depth, embeddings, training parameters, and inference settings
- **Interactive Text Generation**: Accepts user-defined prompts and generates coherent Shakespearean-style output

---

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/verseGPT.git
cd verseGPT
```

### **2. Set Up a Virtual Environment (Optional)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Verify PyTorch Installation**
Ensure PyTorch is correctly installed with CUDA support (if using a GPU):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If using a GPU, the output should be `True`. Otherwise, the model will default to CPU mode.

---

## **Training the Model**
Train verseGPT on the tiny_shakespeare.txt dataset:
```bash
python train.py
```
The training script will:
- Load Shakespearean text and tokenize it
- Train a small GPT model on the dataset
- Save the best-performing model in the `out/` directory

### **Training Parameters**
Modify `train.py` to tweak key parameters:
- `n_embd`: Model embedding size
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `batch_size`: Training batch size
- `learning_rate`: Model learning rate

For GPU users, training will run with AMP for better performance.

---

## **Generating Shakespearean Verse**
Once trained, generate text using:
```bash
python generate.py --checkpoint out/verseGPT_best.pth --prompt "To be, or not to be" --max_tokens 200
```
### **Arguments:**
- `--checkpoint`: Path to the trained model checkpoint
- `--prompt`: The starting phrase for text generation
- `--max_tokens`: Maximum number of tokens to generate

---

## **Example Output**
#### Prompt
```text
Forsooth, artificial intelligence hath not
```
#### Generated Text
```text
Forsooth, artificial intelligence hath not scarr'd the cause:
The second ere I saw the cause to learn
The more, by your good report.

GREMIO:
And more:
And more are welcome, Signior Lucentio.

BAPTISTA:
Away with the dotard! to the gaol with him!

VINCENTIO:
Thus strangers may be hailed and abused: O
monstrous villain!

BIONDELLO:
O! we are spoiled and--yonder he is: deny him,
forswear him, or else we are all undone.

VINCENTIO:
Lives my sweet son?

BIANCA:
Pardon, dear father.

BAPTISTA:
How hast thou offended?
Where is Lucentio?

LUCENTIO:
Here's Lucentio,
```
*(Results will vary depending on training progress)*

## **Potential Improvements**
- Extending beyond Shakespeare with larger collections of Elizabethan verse
- Fine-tuning model hyperparameters to generate more convincing Shakespearean text
- Adding beam search or top-p sampling for better text generation

---

## **Acknowledgments**
- Based on nanoGPT and inspired by [Andrej Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT)
- Uses PyTorch for model training and inference

