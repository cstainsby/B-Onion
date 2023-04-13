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
def clean_res_str(res_str: str):
    cleaned_res_str = res_str

    # replace all newlines and tabs with spaces
    chars_to_spaces = ['\n', '\t']
    for char in chars_to_spaces:
        cleaned_res_str = cleaned_res_str.replace(char, " ")

    # remove all of 
    chars_to_remove = ['\"']
    for char in chars_to_remove:
        cleaned_res_str = cleaned_res_str.replace(char, " ")
    print("clean text" + cleaned_res_str)
    return cleaned_res_str


def get_text_from_openai_res(res) -> str:
    num_choices = len(res["choices"])
    text = str(res["choices"][random.randint(0, num_choices - 1)]["text"])

    # clean data 
    clean_text = clean_res_str(text)

    return clean_text

def get_edition_items_from_openai_res_text(res_text: str) -> dict:
    """Gets all edition contents from an openai res

        randomly selects which openai choice to return
        
        returns dictionary with edition contents
    """
    edition_dict = {}
    edition_dict["title"] = ""
    edition_dict["content"] = ""
    
    title_pattern = r"title:(\s|\S)*content:"
    content_pattern = r"content:(\s|\S)*"

    title_search_obj = re.search(title_pattern, res_text)
    content_search_obj = re.search(content_pattern, res_text)

    # define the lengths of the tags included in the openai res 
    #   to make code more readable
    content_tag_len = 8
    title_tag_len = 6

    if title_search_obj:
        extracted_title = title_search_obj.group(0)
        # remove "title:" and "content:" tags
        trimmed_title = extracted_title[title_tag_len:len(extracted_title)-content_tag_len]
        edition_dict["title"] = trimmed_title

    if content_search_obj:
        extracted_content = content_search_obj.group(0)
        # trim "content:" tag at end
        trimmed_content = extracted_content[content_tag_len:]
        edition_dict["content"] = trimmed_content
        
    return edition_dict