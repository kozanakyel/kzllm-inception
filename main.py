import torch
from data.verdict_data_loader import get_verdict_text_data
from preprocessing.dataloader_process import create_dataloader_v1


### data preprocessing
torch.manual_seed(123)

verdict_raw_data = get_verdict_text_data()
max_length = 4

dataloader = create_dataloader_v1(
    verdict_raw_data,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False,
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("Inputs shape:\n", inputs.shape)

# text embeddings
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


# absolute positional embedding
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# actual embedding vectors
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
