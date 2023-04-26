import re 
from collections import Counter

def parse_class_label_from_AITA_comment(comment: str):
    """Each comment from this subreddit should have a class label within it
    
    RETURNS: class label in undercase or None if there was none"""

    pattern = r"(^|\b)([yY][tT][aA]|[yY][wW][bB][tT][aA]|[nN][tT][aA]|[yY][wW][nN][bB][tT][aA]|[eE][sS][hH]|[nN][aA][hH]|[iI][nN][fF][oO])(\b|$)"

    matches = re.findall(pattern, comment)

    # extract class label from tuple
    matches = [str(match[1]).lower() for match in matches]

    # in case there are multiple classifications, I will do a most common vote, if a tie -> the first inst
    if len(matches) > 0:
        occurence_count = Counter(matches)
        return occurence_count.most_common(1)[0][0]
    else:
        return None
