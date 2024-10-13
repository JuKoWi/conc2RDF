from conc2RDF import (
    parse_the_arg,
    do_the_job,
    Analyzer,
    Directory,
    DataSetFromList

)
from simple import simple
from run_several_samples import multi

import torch

def main():
    args = parse_the_arg()
    if args.m:
        multi()
    elif args.s:
        simple()
    elif args.i is not None:
        if args.i == "default":
            do_the_job()
        else:
            do_the_job(args.i)
    elif args.ad:
        newdir = Directory(args.ad)
        newset = DataSetFromList(newdir.get_relevant_files())
        model = torch.load("./model.pth", weights_only=False)
        my_analyzer = Analyzer(model, newset)
        my_analyzer.get_dashboard()
    elif args.ap:
        newdir = Directory(args.ap)
        newset = DataSetFromList(newdir.get_relevant_files())
        model = torch.load("./model.pth", weights_only=False)
        my_analyzer = Analyzer(model, newset)
        my_analyzer.get_dashboard()
        
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
