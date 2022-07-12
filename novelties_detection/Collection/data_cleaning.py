from html2text import HTML2Text
from bs4 import BeautifulSoup
import re


def extract_text(articlePage, tags, clean = False) -> str:
    """

    @rtype: basestring
    """
    h = HTML2Text()
    h.ignore_links = True
    if clean:
        text = cleanHTML(articlePage , tags)
    else:
        text = str(articlePage)
    text = h.handle(text)  # Get only the text in the tag article
    text = re.sub("\n|#", " ", text)  # Clean the text, cette ligne ne fonctionne plus et détruit le code
    text = re.sub("_|>|<|!\[.*?\]|\(https:.*?\)", "", text)
    return text


def cleanHTML(page , tags_to_remove):

    soup = BeautifulSoup(page, "lxml")  # Parse HTML
    page = soup.find("article")
    for tag in tags_to_remove:
        try:
            if "id" in tag.keys():
                page.find(id=tag["id"]).decompose()
            elif "class" in tag.keys():
                elements = page.find_all(tag["tag"], class_=tag["class"])
                for i, element in enumerate(elements):
                    element.decompose()
            else:
                elements = page.find_all(tag["tag"])
                for element in elements:
                    element.decompose()
        except AttributeError as e:
            pass
        except Exception as e:
            pass
    return str(page)