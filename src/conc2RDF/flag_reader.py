"""Define flags to use in main.py script."""

import argparse


def parse_the_arg():
    """Parse command-line arguments for the main script.

    This function sets up the argument parser and defines the available command-line options.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
            - i (str): Name of the TOML file with input information. Defaults to "default" if not provided.
            - m (bool): Flag indicating if multiple samples should be run; runs the run_several_samples.py script if set.
            - ad (bool): Flag indicating if the dashboard for the last training run should be retrieved.
            - ap (str): Optional argument to show the predictions.
            - ae (str): Optional argument to show the errors.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        nargs="?",
        const="default",
        help="Name of toml with input information",
    )
    parser.add_argument(
        "-m",
        action="store_true",
        help="m for multiple: Run the run_several_samples.py script",
    )
    parser.add_argument(
        "-ad", action="store_true", help="get dashboard for last training run"
    )
    parser.add_argument("-ap", help="show the predictions")
    parser.add_argument("-ae", help="show the errors")
    return parser.parse_args()
