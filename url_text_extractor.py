import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url: str) -> str:
    """
    Extracts and cleans text content from a URL using BeautifulSoup.
    Raises an exception if URL fetch fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        return clean_text

    except Exception as e:
        # Raise exception to handle upstream, don't return error text
        raise RuntimeError(f"Error fetching URL {url}: {e}")
