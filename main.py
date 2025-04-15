from data.verdict_data_loader import get_verdict_text_data
from preprocessing.simple_tokenizer_v1 import SimpleTokenizerV1
from preprocessing.text_splitter import text_splitter

verdict_raw_data = get_verdict_text_data()
preprocessed = text_splitter(verdict_raw_data)
all_words = sorted(set(preprocessed))

vocab = {token: index for index, token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
