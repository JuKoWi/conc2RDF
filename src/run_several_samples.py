"""Where is the best place to ensure that the tensors in RdfDataSet have dtype=torch.float
TODO : Where should I keep the information of the r-values to which the rdf-output values are assigned? Would make sense to me to keep it in the RdfDataSet class
Do I have to add a .to(device) somewhere to actually use the gpu?

flags:
    -p path to data containing directory
"""

from conc2RDF import (
    DataSetFromList,
    Directory,
    NeuralNetwork,
    get_input_args,
)


def main():
    """To get better results with the same hyperparameters the model is initialized several times."""
    arg_dict = get_input_args()
    newdir = Directory(arg_dict["dir_path"])
    newset = DataSetFromList(newdir.get_relevant_files())
    train_conc = [10.0, 30.0, 50.0, 70.0, 90.0]
    test_conc = [20.0, 40.0, 60.0, 80.0, 100.0]
    train_data = newset.get_subset_from_list(newset.get_indices(train_conc))
    test_data = newset.get_subset_from_list(newset.get_indices(test_conc))
    num_runs = 20
    best_val_loss = float("inf")

    for run in range(num_runs):
        model = NeuralNetwork(train_data.get_output_size(), num_neurons=[50, 50, 50], lr=0.001)

        model.train_network(train_data, test_data, 5000)

        val_loss = model.val_losses[-1]
        print(f"Validation Loss for run {run+1}: {val_loss:.2e}")

        # Save the model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model()
    print(f"Best validation loss: {best_val_loss:.2e}")


if __name__ == "__main__":
    main()
