import requests
from bs4 import BeautifulSoup

URLS = [
    "https://scikit-learn.org/stable/getting_started.html",
    "https://scikit-learn.org/stable/modules/cross_validation.html",
    "https://scikit-learn.org/stable/model_selection.html",
]


def load_web_page(url):
    if url.strip() is None:
        raise ValueError("Url is empty !")

    response = requests.get(
        url, timeout=30, headers={"User-Agent": "EducationalRAGEvalBot/1.0"}
    )

    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main")
    if main is None:
        main = soup.find("body")

    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    clean_text = "\n".join(lines)

    return clean_text


if __name__ == "__main__":

    for url in URLS:
        text = load_web_page(url)

        print("=" * 60)
        print("SOURCE:", url)
        print("Characters:", len(text))
        print(text[:500])
