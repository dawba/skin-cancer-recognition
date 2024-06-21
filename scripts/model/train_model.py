from itertools import product


def train_model(cnn, param_grid):
    """
    This function performs a grid search to find the best model hyperparameters.\n

    As specified in ModelBuilder class, the hyperparameters are:\n
    - M: Number of blocks of convolutional layers for image data.\n
    - N: Number of convolutional layers per block.\n
    - K: Number of fully connected layers for metadata.\n
    - learning_rate: Learning rate for the Adam optimizer.\n
    - batch_size: Number of samples per gradient update.\n
    - epochs: Number of epochs to train the model.\n

    :param cnn: NeuralNetworkCNN object
    :param param_grid: Dictionary with hyperparameters to be tested
    :return: best model: best_model, best hyperparameters: best_params, training history: history
    """


    # Convert grid to list of dictionaries
    param_combinations = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]

    best_model, best_params, history = cnn.grid_search(param_combinations)

    return best_model, best_params, history

