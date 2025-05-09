import pytest
import tiktoken
import torch

from core.settings import GPT_CONFIG_124M
from models.dummy_gpt_model import DummyGPTModel
from models.layer_normalization import LayerNorm


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
