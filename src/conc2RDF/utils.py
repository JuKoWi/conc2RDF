"""Potentially useful functions."""


def merge_dictionaries(default_dict: dict, custom_dict: dict) -> dict:
    """Nested dictionaries must be merged to use default_config.toml for parameters not provided in custom .toml."""
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
