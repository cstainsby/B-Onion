import re

def scan_for_edition_tag_ids(prompt: str) -> list:
    pattern = r'@edition\d+'
    # Use re.findall to find all occurrences of the pattern in the string
    matches = re.findall(pattern, prompt)
    # chop '@edition' part off to get the id
    edition_ids = [int(tag[8:]) for tag in matches]
    return edition_ids

def get_included_editions(editions: list, edition_ids: list) -> dict:
    included_editions = {}
    for index in edition_ids:
        if index > 0 and index <= len(editions):
            included_editions[index] = editions[index - 1]
    return included_editions

def editions_to_strs(editions_dict: dict) -> list:
    stringified_editions = []
    for id, edition in editions_dict.items():
        print("edition", id, edition)
        stringified_editions.append(
            """Edition:
                id: {id}
                title: {title}
                content: {content}
            """.format(id=id, title=edition["title"], content=edition["content"])
        )
    return stringified_editions

def formated_req_edition_str(stringified_editions: list) -> str:
    formated_str = ""
    for string_ed in stringified_editions:
        formated_str = formated_str + string_ed + "\n"
    return formated_str