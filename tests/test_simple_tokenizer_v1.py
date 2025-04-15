import pytest
from preprocessing.simple_tokenizer_v1 import SimpleTokenizerV1


@pytest.fixture
def sample_vocab():
    return {
        "hello": 0,
        "world": 1,
        "how": 2,
        "are": 3,
        "you": 4,
        "today": 5,
        "?": 6,
        ",": 7,
        ".": 8,
        "!": 9,
        "<|endoftext|>": 10,
        "<|unk|>": 11,
    }


@pytest.fixture
def tokenizer(sample_vocab) -> SimpleTokenizerV1:
    return SimpleTokenizerV1(sample_vocab)


def test_tokenizer_initialization(tokenizer: SimpleTokenizerV1, sample_vocab: dict):
    """Test that the tokenizer is initialized correctly with the vocabulary"""
    assert tokenizer.str_to_int == sample_vocab
    assert tokenizer.int_to_str == {v: k for k, v in sample_vocab.items()}


def test_encode_simple_text(tokenizer: SimpleTokenizerV1):
    """Test encoding a simple text string"""
    text = "hello world"
    expected_ids = [0, 1]
    assert tokenizer.encode(text) == expected_ids


def test_encode_with_punctuation(tokenizer: SimpleTokenizerV1):
    """Test encoding text with punctuation"""
    text = "hello, world!"
    expected_ids = [0, 7, 1, 9]
    assert tokenizer.encode(text) == expected_ids


def test_encode_unknown_tokens(tokenizer: SimpleTokenizerV1):
    """Test encoding with unknown tokens should raise KeyError"""
    text = "hello unknown world"
    with pytest.raises(KeyError):
        tokenizer.encode(text)


def test_decode_simple_ids(tokenizer: SimpleTokenizerV1):
    """Test decoding simple IDs"""
    ids = [0, 1]
    expected_text = "hello world"
    assert tokenizer.decode(ids) == expected_text


def test_decode_with_punctuation(tokenizer: SimpleTokenizerV1):
    """Test decoding IDs with punctuation"""
    ids = [0, 7, 1, 9]
    expected_text = "hello, world!"
    assert tokenizer.decode(ids) == expected_text


def test_encode_decode_roundtrip(tokenizer: SimpleTokenizerV1):
    """Test that encoding and then decoding returns the original text"""
    text = "hello, world! how are you today?"
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    assert decoded_text == text


def test_decode_unknown_ids(tokenizer: SimpleTokenizerV1):
    """Test decoding with unknown IDs should raise KeyError"""
    ids = [0, 1, 999]  # 999 is not in the vocabulary
    with pytest.raises(KeyError):
        tokenizer.decode(ids)


def test_encode_with_endoftext(tokenizer: SimpleTokenizerV1):
    """Test encoding with endoftext token"""
    text = "hello world <|endoftext|>"
    expected_ids = [0, 1, 10]
    assert tokenizer.encode(text) == expected_ids


def test_decode_with_endoftext(tokenizer: SimpleTokenizerV1):
    """Test decoding with endoftext token"""
    ids = [0, 1, 10]  # hello world <|endoftext|>
    expected_text = "hello world <|endoftext|>"
    assert tokenizer.decode(ids) == expected_text


def test_encode_with_unk(tokenizer: SimpleTokenizerV1):
    """Test encoding with unk token"""
    text = "hello world <|unk|>"
    expected_ids = [0, 1, 11]
    assert tokenizer.encode(text) == expected_ids


def test_decode_with_unk(tokenizer: SimpleTokenizerV1):
    """Test decoding with unk token"""
    ids = [0, 1, 11]  # hello world <|unk|>
    expected_text = "hello world <|unk|>"
    assert tokenizer.decode(ids) == expected_text


def test_encode_with_both_special_tokens(tokenizer: SimpleTokenizerV1):
    """Test encoding with both special tokens"""
    text = "hello <|unk|> world <|endoftext|>"
    expected_ids = [0, 11, 1, 10]
    assert tokenizer.encode(text) == expected_ids


def test_decode_with_both_special_tokens(tokenizer: SimpleTokenizerV1):
    """Test decoding with both special tokens"""
    ids = [0, 11, 1, 10]  # hello <|unk|> world <|endoftext|>
    expected_text = "hello <|unk|> world <|endoftext|>"
    assert tokenizer.decode(ids) == expected_text
