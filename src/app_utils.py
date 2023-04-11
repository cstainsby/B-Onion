import re
import random

from praw_instance import PrawInstance


# -----------------------------------------------------------------------
#       Scan Functions
# -----------------------------------------------------------------------
def scan_for_edition_tag_ids(prompt: str) -> list:
    pattern = r'@edition\d+'
    # Use re.findall to find all occurrences of the pattern in the string
    matches = re.findall(pattern, prompt)
    # chop '@edition' part off to get the id
    edition_ids = [int(tag[8:]) for tag in matches]
    return edition_ids

def scan_for_reddit_tags(prompt: str) -> list:
    pattern = r"@reddit-[a-zA-Z0-9]{7}"
    # Use re.findall to find all occurrences of the pattern in the string
    matches = re.findall(pattern, prompt)
    # chop '@reddit-' part off to get the id
    reddit_ids = [tag[8:] for tag in matches]
    return reddit_ids


# -----------------------------------------------------------------------
#       Get by tag id functions
# -----------------------------------------------------------------------
def get_included_editions(editions: list, edition_ids: list) -> dict:
    included_editions = {}
    for index in edition_ids:
        if index > 0 and index <= len(editions):
            included_editions[index] = editions[index - 1]
    return included_editions


# -----------------------------------------------------------------------
#       Tagged content to string representation 
#   These functions will all convert a given tagged items contents to
#   an Edition which is 
# -----------------------------------------------------------------------
def make_edition_str(id: str, title: str, content: str) -> str:
    """Helper function for each of the types to convert their content 
        to a standardized format"""
    edition = """Edition:
                id: {id}
                title: {title}
                content: {content}
            """.format(id=id, title=title, content=content)
    return edition

def editions_to_strs(editions_dict: dict) -> list:
    stringified_editions = []
    for id, edition in editions_dict.items():
        stringified_editions.append(
            make_edition_str(
                id=id,
                title=edition["title"],
                content=edition["content"]
            )
        )
    return stringified_editions

def reddit_post_to_edition_str(post_obj) -> list:
    reddit_edition_str = make_edition_str(
        id=post_obj,
        title=post_obj.title,
        content=post_obj.selftext
    )
    return reddit_edition_str

# -----------------------------------------------------------------------
#       format tagged content to request
# -----------------------------------------------------------------------
def formated_req_edition_str(stringified_editions: list) -> str:
    formated_str = ""
    for string_ed in stringified_editions:
        formated_str = formated_str + string_ed + "\n"
    return formated_str

# -----------------------------------------------------------------------
#       
# -----------------------------------------------------------------------
def get_text_from_openai_res(res) -> str:
    num_choices = len(res["choices"])
    text = str(res["choices"][random.randint(0, num_choices - 1)]["text"])

    # clean data 
    text.replace("\n", " ")

    return text

def get_edition_items_from_openai_res_text(res_text: str) -> dict:
    """Gets all edition contents from an openai res

        randomly selects which openai choice to return
        
        returns dictionary with edition contents
    """
    edition_dict = {}
    
    titles = re.findall(r'title:\s*"([^"]+)"', res_text)
    contents = re.findall(r'content:\s*"([^"]+)"', res_text)

    if len(titles) > 0 and len(contents) > 0:
        edition_dict["title"] = titles[0]
        edition_dict["content"] = contents[0]
    else:
        edition_dict["title"] = ""
        edition_dict["content"] = ""
    return edition_dict