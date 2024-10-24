"""Potentially useful functions."""


def merge_dictionaries(default_dict: dict, custom_dict: dict) -> dict:
    """Merge two dictionaries, with values from the custom dictionary overriding defaults.

    This function recursively merges two dictionaries. If a key exists in both dictionaries 
    and its value is a dictionary in both, they are merged recursively. If the key exists only 
    in the default dictionary, the default value is retained.

    Args:
        default_dict (dict): The dictionary containing default values.
        custom_dict (dict): The dictionary containing custom values to override defaults.

    Returns:
        dict: A new dictionary with merged values from the default and custom dictionaries. 
              The custom dictionary takes precedence for any overlapping keys.
    """
    merged_dict = {}

    for key, default_value in default_dict.items():
        custom_value = custom_dict.get(key, None)

        if isinstance(default_value, dict) and isinstance(custom_value, dict):
            # Recursively merge nested dictionaries
            merged_dict[key] = merge_dictionaries(default_value, custom_value)
        else:
            # If no custom value, or custom value isn't a dictionary, use default or custom
            merged_dict[key] = custom_value if key in custom_dict else default_value

    return merged_dict
