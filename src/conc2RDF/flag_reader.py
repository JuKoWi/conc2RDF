import argparse


def parse_the_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "-p", help="Path to dataset")
    parser.add_argument("--i", "-i", nargs="?", const = "default", help="Name of toml with input information")
    parser.add_argument("-m", "--m", action="store_true", help=" m for multiple: Run the run_several_samples.py script")
    parser.add_argument("-s", "--s", action="store_true", help="s for single: Run the simple.py script")
    parser.add_argument("-ad", action="store_true", help="get dashboard for last training run")
    parser.add_argument("-ap", help="show the predictions")
    args = parser.parse_args()
    return args




# def parse_arguments():  # More descriptive function name
#     parser = argparse.ArgumentParser(description="Script to run different training/evaluation tasks.")  # Add description
#     parser.add_argument("--path", "-p", required=True, help="Path to dataset")  # Make dataset path required
#     parser.add_argument("--input", "-i", default="default", help="Name of TOML file with input information (default: 'default')")  # Improve help message
#     parser.add_argument("--multiple", "-m", action="store_true", help="Run the run_several_samples.py script")
#     parser.add_argument("--single", "-s", action="store_true", help="Run the simple.py script")
#     parser.add_argument("--dashboard", "-ad", action="store_true", help="Get dashboard for the last training run")
#     parser.add_argument("--predictions", "-ap", help="Show the predictions")
#     args = parser.parse_args()
#     return args