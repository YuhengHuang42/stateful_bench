from functools import reduce
from operator import getitem
import json

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

def get_added_changes(old_book, new_book):
    changes = {}
    for k, v in new_book.items():
        if k not in old_book:
            # New key
            changes[k] = v
        else:
            assert v >= old_book[k], f"The number of occurence of {k} is decreased from {old_book[k]} to {v}."
            # Existing key with increased value
            if v > old_book[k]:
                changes[k] = v - old_book[k]
    return changes

def write_jsonl(output_path, data):
    with open(output_path, "w") as ifile:
        for entry in data:
            json_line = json.dumps(entry)
            ifile.write(json_line + '\n')

def merge_dicts(*dicts):
    """
    Merge multiple dictionaries into a single dictionary.
    If there are duplicate keys, the value from the later dictionary will override the earlier one.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        dict: Merged dictionary
    """
    result = {}
    for d in dicts:
        if d is not None:
            result.update(d)
    return result
