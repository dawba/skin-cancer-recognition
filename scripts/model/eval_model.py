import numpy as np
from scripts.combine_inputs import combine_inputs
from sklearn.metrics import accuracy_score, classification_report

def eval_model(model, test_images, test_metadata, test_labels):
    """
    Evaluate the performance of the trained model on the test data.

    Parameters:
    model: Trained keras model.
    test_images (np.ndarray): Array of test images.
    test_metadata (np.ndarray): Array of test metadata.
    test_labels (np.ndarray): Array of test labels.

    Returns:
    predicted_labels: Array of predicted labels by the model.
    """
    print("Evaluating model...")

    # Combine images and metadata
    combined_input = combine_inputs(test_images, test_metadata)

    # Predict labels for test data
    predictions = model.predict([test_images, test_metadata])
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)

    # Generate classification report
    class_report = classification_report(test_labels, predicted_labels)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)

    return predicted_labels, predictions, accuracy


test_images, test_metadata = test_X
test_labels, _ = encode_metadata(test_metadata, encoder=encoder, fit=False)

predicted_labels, predictions, accuracy = eval_model(best_model, test_images, test_labels, test_Y)