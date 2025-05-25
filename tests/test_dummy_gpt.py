import pytest
import tiktoken
import torch
import torch.nn as nn
from models.gelu import GELU, FeedForward

from core.settings import GPT_CONFIG_124M
from models.dummy_gpt_model import DummyGPTModel
from models.gpt_model import GPTModel
from models.layer_normalization import LayerNorm
from models.transformer_block import TransformerBlock


def test_bacth_dummy():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print("batch", batch)
    
    torch.manual_seed(123)
    model = DummyGPTModel(cfg=GPT_CONFIG_124M)
    logits = model(batch)
    print("logits", logits.shape)
    print("logits", logits)
    
def test_seqential_process():
    torch.manual_seed(123)
    batch_example = torch.randn(2,5)
    layer = torch.nn.Sequential(torch.nn.Linear(5,6), torch.nn.ReLU())
    out = layer(batch_example)
    print("out", out.shape)
    print("out", out)
    
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("mean", mean)
    print("var", var)
    
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    torch.set_printoptions(sci_mode=False)
    print(f"normalized layer outputs: {out_norm}")
    print("mean", mean)
    print("var", var)
    
    # new layer normalization class
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
    print("mean", mean)
    print("var", var)
    
def test_gelu():
    import matplotlib.pyplot as plt
    gelu, relu = GELU(), nn.ReLU()
    
    x = torch.linspace(-3, 3, 1000)
    y_gelu = gelu(x)
    y_relu = relu(x)
    plt.figure(figsize=(8,3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def test_feedforward():
    ffn  = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2,3,768)
    out = ffn(x)
    print("FeedForward output shape:", out.shape)
    assert out.shape == (2, 3, 768), "FeedForward output shape mismatch"
    
    
def test_transformer_block():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == (2, 4, 768), "Transformer block output shape mismatch"
    
    
def test_gpt_model():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print("batch", batch)
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)
    
    model.eval()
    


