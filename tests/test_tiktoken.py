from importlib.metadata import version
import tiktoken

from data.verdict_data_loader import get_verdict_text_data

print(version("tiktoken"))


tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

decoded = tokenizer.decode(integers)
print(decoded)


verdict_raw_data = get_verdict_text_data()
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(verdict_raw_data)
print(f"len of enc_text: {len(enc_text)}, part of 10 tokens: {enc_text[:10]}")


def test_tiktoken_predict():
    enc_sample = enc_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1 : context_size + 1]
    print(f"x: {x}")
    print(f"y:     {y}")

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
