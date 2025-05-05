import pytest
import torch


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


def softmax_naive(x):
    # all values always positive with the softmax function and between 0 and 1
    return torch.exp(x) / torch.exp(x).sum(dim=0)


# implement the intermediate values "w" as a attention scores
# use truncated values from vectors like 0.87 to 0.8


# dot procut multiplying 2 vector then summing all values return a scalar result
# higher w attention score measn higher similarity
def test_calculate_w():
    query = inputs[1]  # x^2
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)

    # normalization for w scores
    # we obtain a attention weights !!!! x to w to a
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print(f"attention weights: {attn_weights_2_tmp}")
    print(f"sum: {attn_weights_2_tmp.sum()}")

    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print(f"attention weights naive: {attn_weights_2_naive}")
    print(f"sum: {attn_weights_2_naive.sum()}")


def test_calculate_context_vec():  # ...  to z
    # x->w->a->z
    query = inputs[1]  # x^2

    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)

    attn_weights_2_naive = softmax_naive(attn_scores_2)

    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2_naive[i] * x_i

    print(f"context vector: {context_vec_2}")


def test_calculate_context_vectors_z():
    # attn_scores = torch.empty(6, 6)
    # for i, x_i in enumerate(inputs):
    #     for j, x_j in enumerate(inputs):
    #         attn_scores[i, j] = torch.dot(x_i, x_j)

    # performance based on matrix multiplication
    attn_scores = inputs @ inputs.T  # x_i @ x_j = w_i,j
    print(attn_scores)  # w -> attention scores

    attn_weights = torch.softmax(attn_scores, dim=1)  # w->a  //softmax on the rows
    print(attn_weights)  # a -> attention weights
    
    print("All row sums:", attn_weights.sum(dim=-1))
    
    all_context_vecs = attn_weights @ inputs  # a->z //softmax on the rows
    print(all_context_vecs)  # z -> context vectors
