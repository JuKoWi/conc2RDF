import torch
import torch.optim as optim

from conc2RDF import (
    Analyzer,
    Callbacks,
    DataSetFromList,
    Directory,
    EarlyStoppingCallback,
    LRScheduler,
    NeuralNetwork,
    load_toml,
    parse_the_arg,
    validate_input,
)


def main():
    """Read arguments and perform training or analysis.

    This function parses command-line arguments, reads configuration settings,
    performs training of a neural network, or analyzes results based on the provided flags.
    It handles model training, early stopping, learning rate scheduling, and
    generating relevant plots or metrics.

    The following command-line arguments are supported:
        -i: Input TOML configuration file for training.
        -m: Flag to run the script with multiple samples.
        -ad: Flag to get the dashboard for the last training run.
        -ap: Path to the dataset to show predictions.
        -ae: Path to the dataset to show errors.
    """
    args = parse_the_arg()
    if args.i is not None:
        """Read configuration, perform training and save the best result."""
        filepath = args.i
        config = load_toml(filepath)
        validate_input(config)
        job_dir = Directory(config.data.dirpath)
        jobset = DataSetFromList(job_dir.get_relevant_files())
        train_conc = config.learn.train_selection
        test_conc = config.learn.test_selection
        train_data = jobset.get_subset_from_list(jobset.get_indices(train_conc))
        test_data = jobset.get_subset_from_list(jobset.get_indices(test_conc))
        num_runs = config.learn.num_runs

        best_val_loss = float("inf")
        """Do several initializations of the model for different start parameters."""
        for run in range(num_runs):
            model = NeuralNetwork(
                train_data.get_output_size(),
                config.nn.num_neurons,
            )
            optimizer = optim.Adam(
                model.parameters(), lr=config.nn.optimizer.learning_rate
            )
            """Set up callbacks"""
            early_stop = Callbacks()
            if config.learn.stopping.is_on:
                early_stop = EarlyStoppingCallback(
                    patience=config.learn.stopping.patience,
                    min_delta=config.learn.stopping.min_delta,
                )
            scheduler = Callbacks()
            if config.learn.scheduler.is_on:
                scheduler_setup = getattr(
                    optim.lr_scheduler,
                    config.learn.scheduler.type,
                )(
                    optimizer,
                    mode=config.learn.scheduler.mode,
                    factor=config.learn.scheduler.factor,
                    patience=config.learn.scheduler.patience,
                    verbose=config.learn.scheduler.verbose,
                )
                scheduler = LRScheduler(
                    scheduler_setup
                )  # takes optim.lr_scheduler object as input
            model.train_network(
                train_data,
                test_data,
                epochs=config.learn.epochs,
                print_progress=config.learn.print,
                callbacks=[early_stop, scheduler],
                optimizer=optimizer,
            )
            val_loss = model.val_losses[-1]
            print(f"Validation Loss for run {run + 1}: {val_loss:.2e}")
            """Save the model if it has the best validation loss"""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_model()
        print(f"Best validation loss: {best_val_loss:.2e}")

    elif args.ad:
        model = torch.load("./model.pth", weights_only=False)
        my_analyzer = Analyzer(model)
        my_analyzer.get_dashboard()

    elif args.ap:
        newdir = Directory(args.ap)
        newset = DataSetFromList(newdir.get_relevant_files())
        model = torch.load("./model.pth", weights_only=False)
        my_analyzer = Analyzer(model)
        my_analyzer.show_predictions(newset)

    elif args.ae:  # uses dataset but just out of pure programming laziness
        newdir = Directory(args.ae)
        newset = DataSetFromList(newdir.get_relevant_files())
        model = torch.load("./model.pth", weights_only=False)
        my_analyzer = Analyzer(model)
        my_analyzer.show_errors(newset)

    if not any(vars(args).values()):
        print(
            "No flags were provided. Please provide at least one flag.\n",
            "\t-i <filename.toml> to provide an input file that specifies the job (preferred)\n",
            "\t-ap <path/to/dataset> to get predictions for model.pth\n",
            "\t-ad to get dashboard for the last training run of model.pth\n",
            "\t-ae <path/to/dataset> to get MSE and MAE for model.pth\n",
        )


if __name__ == "__main__":
    main()
