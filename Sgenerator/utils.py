from functools import reduce
from operator import getitem

def get_nested_value(dct, key_list):
    """Safely get nested value from dictionary using list of keys"""
    try:
        return reduce(getitem, key_list, dct)
    except (KeyError, TypeError):
        return None  # Or raise appropriate exception