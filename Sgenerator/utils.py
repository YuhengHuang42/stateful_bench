from functools import reduce
from operator import getitem

def get_nested_value(dct: dict, key_list: list):
    """Safely get nested value from dictionary using list of keys"""
    try:
        return reduce(getitem, key_list, dct)
    except (KeyError, TypeError):
        return None  # Or raise appropriate exception

def get_nested_path_string(dct_str: str, key_list: list):
    """Return string representation of nested dictionary access path"""
    path = dct_str
    for key in key_list:
        if isinstance(key, str):
            path += f'["{key}"]'
        else:
            path += f'[{key}]'
    return path