import torch
import tiktoken

from core.settings import GPT_CONFIG_124M
from models.gpt_model import GPTModel, generate_text_simple


def text_to_token_ids(text, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer: tiktoken.Encoding):
    # very interesting this decode always not possible to utf-e, you should control it
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def test_tokenizers():

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    inputs = torch.tensor(
        [[16833, 3626, 6100], [40, 1107, 588]]  # ["every effort moves",
    )  # "I really like"]

    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)
    
    
