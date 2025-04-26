import pytest
import torch


def test_embedding():
    # for mebedding we use use the random initilizaedc values
    in_ids = torch.tensor([2,3,5,1])
    vocab_size = 6
    output_dim = 3
    
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    
    print(embedding_layer(in_ids))
    
def test_embedding_v2():
    # for mebedding we use use the random initilizaedc values
    in_ids = torch.tensor([2,3,5,1])
    vocab_size = 50257
    output_dim = 256
    
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    
    print(embedding_layer(in_ids))