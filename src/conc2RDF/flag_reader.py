"""Define flags to use in main.py script."""

import argparse


def parse_the_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "-p", help="Path to dataset")
    parser.add_argument(
        "--i",
        "-i",
        nargs="?",
        const="default",
        help="Name of toml with input information",
    )
    parser.add_argument(
        "-m",
        "--m",
        action="store_true",
        help=" m for multiple: Run the run_several_samples.py script",
    )
    parser.add_argument(
        "-s", "--s", action="store_true", help="s for single: Run the simple.py script"
    )
    parser.add_argument(
        "-ad", action="store_true", help="get dashboard for last training run"
    )
    parser.add_argument("-ap", help="show the predictions")
    parser.add_argument("-ae", help="show the errors")
    return parser.parse_args()
