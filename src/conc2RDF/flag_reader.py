import argparse


def parse_the_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "-p", help="Path to dataset")
    parser.add_argument("--i", "-i", nargs="?", const = "default", help="Name of toml with input information")
    parser.add_argument("-m", "--m", action="store_true", help=" m for multiple: Run the run_several_samples.py script")
    parser.add_argument("-s", "--s", action="store_true", help="s for single: Run the simple.py script")
    parser.add_argument("-ad", help="get dashboard for last raining run")
    parser.add_argument("-ap", help="show the predictions")
    args = parser.parse_args()
    return args
