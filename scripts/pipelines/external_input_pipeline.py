from scripts.utils.load_data import load_data
from scripts.utils.preprocess_data import preprocess_data
from scripts.utils.encode_metadata import encode_metadata
from scripts.utils.combine_inputs import combine_inputs
from scripts.utils.split_data import split_data
from sklearn.metrics import accuracy_score, classification_report

from joblib import load
import numpy as np

example_model_path = '../model/cnn_final_0.8452578783035278.pkl'
example_metadata_path = '../../data/external_pipeline_data/metadata/metadata.csv'
example_images_path = '../../data/external_pipeline_data/images'
example_destination_folder = '../../data/external_pipeline_data/all_images'

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # Adjust based on your dataset's classes


def external_input_pipeline():
    print("This is a pipeline for testing purposes.")
    print("It is meant to test the model on external data.")

    # Get user input for external data paths
    # images_folder = input("Enter the path to the folder containing images: ")
    # metadata_folder = input("Enter the path to the metadata file: ")
    # augmented_folder = input("Enter the path to the folder containing augmented images: ")

    # Load data
    images, metadata = load_data(example_images_path, None, example_destination_folder, example_metadata_path)

    # Preprocess data
    images, metadata = preprocess_data(images, metadata, None, augmentation=True, target_size=(28, 28))

    # Split data but all is validation data
    _, test_X, _, test_Y = split_data(images, metadata, validation_split=1.0)
    images, metadata = test_X

    # Encode metadata
    encoded_metadata, encoder = encode_metadata(metadata, fit=True)

    # Load the model
    model = load(example_model_path)
    print("Model loaded successfully!\n")
    print(model.summary())

    # Use the model to make predictions
    combined_input = combine_inputs(images, encoded_metadata)
    predictions = model.predict([images, encoded_metadata])
    predicted_labels = np.argmax(predictions, axis=1)

    print("Predictions made successfully!\n")
    for i, (label, prediction) in enumerate(zip(predicted_labels, predictions)):
        class_name = class_names[label]
        probability = prediction[label]
        print(f"Image {i + 1}:")
        print(f"Predicted Class: {class_name}")
        print(f"Probability: {probability:.4f}\n")

    # Evaluate predictions
    accuracy = accuracy_score(test_Y, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")


external_input_pipeline()
