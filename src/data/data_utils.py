import re 
from collections import Counter

def most_common_vote(item_list: list):
    """A generic function for finding the most common item in a list
    
    RETURNS:
        most common item, if tie -> the first item in counted list"""
    
    if len(item_list) > 0:
        occurence_count = Counter(item_list)
        return occurence_count.most_common(1)[0][0]
    else:
        return None
    
def get_AITA_most_common_vote(vote_list: list):
    """Wrapper function for most_common_vote for getting the most common vote
    
    It will remove any None's"""
    if None in vote_list:
        vote_list.remove(None)

    return most_common_vote(vote_list)

def parse_class_label_from_AITA_comment(comment: str):
    """Each comment from this subreddit should have a class label within it
    
    RETURNS: class label in undercase or None if there was none"""

    pattern = r"(^|\b)([yY][tT][aA]|[yY][wW][bB][tT][aA]|[nN][tT][aA]|[yY][wW][nN][bB][tT][aA]|[eE][sS][hH]|[nN][aA][hH]|[iI][nN][fF][oO])(\b|$)"

    matches = re.findall(pattern, comment)

    # extract class label from tuple
    matches = [str(match[1]).lower() for match in matches]

    return most_common_vote(matches)
