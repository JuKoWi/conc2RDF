import argparse


def parse_the_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "-p", help="Path to dataset")
    parser.add_argument("--i", "-i", help="Name of toml with input information")
    parser.add_argument("-m", "--m", action="store_true", help=" m for multiple: Run the run_several_samples.py script")
    parser.add_argument("-s", "--s", action="store_true", help="s for single: Run the simple.py script")
    args = parser.parse_args()
    return args