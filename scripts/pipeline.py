from scripts.encode_metadata import encode_metadata
from scripts.load_data import load_data
from scripts.model.build_model import ModelBuilder
from scripts.preprocess_data import preprocess_data
from scripts.split_data import split_data
from scripts.model.neural_network_cnn import NeuralNetworkCNN
from scripts.model.eval_model import eval_model
from scripts.model.train_model import train_model
from scripts.plot_results import plot_results

folder1 = '../data/HAM10000/HAM10000_images_part_1'
folder2 = '../data/HAM10000/HAM10000_images_part_2'
destination_folder = '../data/HAM10000/HAM10000_images'
metadata_file = '../../skin-cancer-recognition/data/HAM10000/HAM10000_metadata.csv'
augmented_folder = '../data/HAM10000/HAM10000_images_augmented'


def run_main_pipeline():

    # 1. Load data
    images, metadata = load_data(folder1, folder2, destination_folder, metadata_file)

    # 2. Preprocess data
    images, metadata = preprocess_data(images, metadata, augmented_folder)

    # 3. Split data
    train_X, test_X, train_Y, test_Y = split_data(images, metadata, validation_split=0.2)
    train_images, train_metadata = train_X

    # 4. Encode metadata
    transformed_train_metadata, encoder = encode_metadata(train_metadata, fit=True)

    # 5, 6. Combine data, build model
    cnn_network = NeuralNetworkCNN(images=train_images, metadata=transformed_train_metadata, labels=train_Y, model_builder=ModelBuilder())

    # 7. Train model
    example_hyperparam_grid = {
        'M': [2],
        'N': [4],
        'K': [2],
        'learning_rate': [1e-3],
        'epochs': [30],
        'batch_size': [128]
    }

    best_model, best_params, history = train_model(cnn_network, example_hyperparam_grid)

    # 8. Evaluate model
    test_images, test_metadata = test_X
    test_labels, _ = encode_metadata(test_metadata, encoder=encoder, fit=False)
    predicted_labels, predictions, _ = eval_model(best_model, test_images, test_labels, test_Y)

    # 9. Plot results
    plot_results(history, test_Y, predicted_labels, predictions)

    # 10. Save model, write params to txt file
    best_model.save('../models/best_model.h5')
    with open('../models/best_params.txt', 'w') as f:
        f.write(str(best_params))


