from conc2RDF import (
    DataSetFromList,
    Directory,
    NeuralNetwork,
    get_input_args,
    parse_the_arg
)
from simple import simple
from run_several_samples import multi

def main():
    args = parse_the_arg()
    if args.m:
        multi()
    elif args.s:
        simple()
    if not any(vars(args).values()):
        print(
            "No flags were provided. Please provide at least one flag.\n",
            "\t-i <filename.toml> to provide an input file that specifies the job (preferred)\n",
            "\t-p <path> to provide the path to a data containing directory\n",
            "\t-m to run multi initial conditions job with default settings; -p flag necessary\n",
            "\t-s to run single job with default settings; -p flag necessary"
        )


if __name__ == "__main__":
    main()
