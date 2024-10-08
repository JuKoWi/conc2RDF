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
    arg_dict = get_input_args()
    newdir = Directory(arg_dict["dir_path"])
    newset = DataSetFromList(newdir.get_relevant_files())
    train_conc = [10.0, 30.0, 50.0, 70.0, 90.0]
    test_conc = [20.0, 40.0, 60.0, 80.0, 100.0]
    train_data = newset.get_subset_from_list(newset.get_indices(train_conc))
    test_data = newset.get_subset_from_list(newset.get_indices(test_conc))
    model = NeuralNetwork(train_data.get_output_size(), lr=0.0001, num_neuron=50)
    print(model.device)
    model.train_network(train_data, test_data, 1500)
    model.save_model()


if __name__ == "__main__":
    main()
