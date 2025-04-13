import re

from data.verdict_data_loader import get_verdict_text_data
from tokenization.text_splitter import text_splitter


def test_split_text():
    verdict = get_verdict_text_data()

    splitted_text = re.split(r'([,.:;?_!"()\']|--|\s)', verdict)

    assert len(splitted_text) > 0, "Splitted text is empty"
    assert len(splitted_text) == 9235, "Splitted text does not match expected length"

    # for whitespcee removed from list

    splitted_text = [text.strip() for text in splitted_text if text.strip()]

    print(f"removed whitespaces length: {len(splitted_text)}")


def test_splitter_text():
    verdict = get_verdict_text_data()

    splitted_text = text_splitter(verdict)

    assert len(splitted_text) > 0, "Splitted text is empty"
    assert len(splitted_text) == 4690, "Splitted text does not match expected length"
    assert (
        len(splitted_text) > 2000
    ), "Splitted text does not match expected length of 2000 greater"
