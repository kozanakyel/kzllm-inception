import urllib.request
import os


def test_data_download():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "the-verdict.txt"
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    urllib.request.urlretrieve(url, file_path)

    assert os.path.exists(file_path), "File was not downloaded"

    with open(file_path, "r", encoding="utf-8") as file:
        verdict = file.read()

    assert len(verdict) > 0, "File is empty"
