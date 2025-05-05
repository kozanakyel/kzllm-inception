import pytest
import torch

from preprocessing.self_attention import CasualAttention, MultiHeadAttention, MultiHeadAttentionWrapper, SelfAttention_v1, SelfAttention_v2

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],
    ]  # step (x^6)
)

torch.manual_seed(123)



def test_intro_variables():
    # normally input and utput embedding size is eqaul but this time we use different sizes
    x_2 = inputs[1]  # x^2
    d_in = inputs.shape[1]  # input embedding size! =>3
    d_out = 2 ## output embedding size! =>2
    
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 3x2
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 3x2
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 3x2
    
    # this weight for trainable for most effective findeing model weights. not the same as attention weights
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(f"query_2: {query_2}")
    
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    
    # attention score w2
    keys_2 = keys[1]  # x^2
    attn_score_22 = query_2.dot(keys_2)  # 2x2 dot product
    print(f"attn_scores_22: {attn_score_22}")
    
    ## all attention score for w2
    attn_scores_2 = query_2 @ keys.T  # 2x2 dot product
    print(f"attn_scores_2: {attn_scores_2}")
    
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)  # 2x2 softmax
    print(f"attn_weights_2: {attn_weights_2}")
    
    context_vec_2 = attn_weights_2 @ values  # 2x2 dot product
    print(f"context_vec_2: {context_vec_2}")

def test_scaled_dot_product_attention():
    torch.manual_seed(789)
    x_2 = inputs[1]  # x^2
    d_in = inputs.shape[1]  # input embedding size! =>3
    d_out = 2 ## output embedding size! =>2
    
    sa_v1 = SelfAttention_v2(d_in, d_out)
    print(sa_v1(inputs))
    
def test_casual_attention():
    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)

    x_2 = inputs[1]  # x^2
    d_in = inputs.shape[1]  # input embedding size! =>3
    d_out = 2 ## output embedding size! =>2
    
    context_length = batch.shape[1]
    ca = CasualAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("CA context_vecs.shape:", context_vecs.shape)
    
    # multiheadattentionwrapper
    torch.manual_seed(123)
    context_length = batch.shape[1] # This is the number of tokens
    d_in, d_out = 3, 2
    
    mha = MultiHeadAttentionWrapper(
        d_in=d_in, d_out=d_out, num_heads=2, context_length=context_length, dropout=0.0
    )
    context_vecs = mha(batch)
    
    print("MHAW context_vecs.shape:", context_vecs.shape)
    
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)