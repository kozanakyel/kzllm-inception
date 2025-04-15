import re


def text_splitter(raw_text):
    """Splits the text into smaller chunks."""
    splitted_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    words = [text.strip() for text in splitted_text if text.strip()]
    return words
