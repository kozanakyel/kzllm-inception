from data.verdict_data_loader import get_verdict_text_data
from preprocessing.text_splitter import text_splitter


def test_text_splitter_empty_input():
    """Test with empty input"""
    assert text_splitter("") == [], "Should return empty list for empty input"


def test_text_splitter_single_word():
    """Test with single word"""
    result = text_splitter("hello")
    assert len(result) == 1
    assert result[0] == "hello"


def test_text_splitter_special_characters():
    """Test handling of special characters"""
    result = text_splitter("hello, world! How are you?")
    assert "," not in result
    assert "!" not in result
    assert "?" not in result


def test_text_splitter_multiple_spaces():
    """Test handling of multiple spaces"""
    result = text_splitter("hello    world")
    assert len(result) == 2
    assert result == ["hello", "world"]


def test_text_splitter_case_sensitivity():
    """Test case sensitivity"""
    result = text_splitter("Hello WORLD")
    assert "Hello" in result
    assert "WORLD" in result


def test_text_splitter_numbers():
    """Test handling of numbers"""
    result = text_splitter("text with 123 numbers")
    assert "123" in result


def test_splitter_performance():
    """Test performance with large text"""
    verdict = get_verdict_text_data()
    splitted_text = text_splitter(verdict)
    assert len(splitted_text) > 0
    assert len(splitted_text) == 3826
