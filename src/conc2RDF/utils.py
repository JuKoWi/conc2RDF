
import sys

import matplotlib.pyplot as plt

from .neural_network import NeuralNetwork

"""some miscellaneous functions"""


def is_intersection_train_test(train_dataset, test_dataset):
    for cons1 in train_dataset.inputs:
        for cons2 in test_dataset.inputs:
            if cons1 == cons2:
                return True
    return False


def get_input_args():
    arg_dict = {}
    for i, arg in enumerate(sys.argv):
        if arg.startswith("-p"):
            arg_dict["dir_path"] = sys.argv[i + 1]
    return arg_dict

def merge_dictionaries(default_dict, custom_dict):
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




