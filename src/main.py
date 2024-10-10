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
    if args.s:
        simple()
        

if __name__ == "__main__":
    main()
