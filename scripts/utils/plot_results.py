from itertools import cycle
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import  confusion_matrix, roc_curve, auc, \
    precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class_names = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]


def plot_history(history):
    """
    Plot the training history.

    Parameters:
    history: History object returned from model.fit()
    """
    # Plotting the accuracy of the model for each epoch
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plotting the loss of the model for each epoch
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


def plot_confusion_matrix(test_labels, predicted_labels):
    """
    Plot the confusion matrix.

    Parameters:
    test_labels: Array of true labels.
    predicted_labels: Array of predicted labels.
    """
    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    return conf_matrix


def plot_prediction_fractions(conf_matrix):
    """
    Plot the fractions of correct and incorrect predictions for each class.

    Parameters:
    conf_matrix: Confusion matrix
    class_names: List of class names
    """

    # Calculate fractions of correct and incorrect predictions
    class_fraction_positive = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    class_fraction_negative = 1 - class_fraction_positive

    # Define colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    # Plot correct prediction fractions
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(class_names)), class_fraction_positive, color=colors)
    plt.title('Correct Prediction Fractions of Classes')
    plt.xlabel('True Label')
    plt.ylabel('Fraction Classified Correctly')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.ylim(0, 1)
    plt.show()

    # Plot incorrect prediction fractions
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(class_names)), class_fraction_negative, color=colors)
    plt.title("Incorrect Prediction Fractions of Classes")
    plt.xlabel('True Label')
    plt.ylabel('Fraction Classified Incorrectly')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.ylim(0, 1)
    plt.show()


def plot_roc_curves(test_labels, predictions):
    """
    Plot ROC curves for each class.

    Parameters:
    test_labels: Array of true labels.
    predictions: Array of predicted probabilities.
    class_names: List of class names.
    """
    lb = LabelBinarizer()
    lb.fit(test_labels)
    test_labels_binary = lb.transform(test_labels)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(test_labels_binary[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'turquoise', 'darkgreen', 'red'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curves(test_labels, predictions):
    """
    Plot Precision-Recall curves for each class.

    Parameters:
    test_labels: Array of true labels.
    predictions: Array of predicted probabilities.
    class_names: List of class names.
    """
    lb = LabelBinarizer()
    lb.fit(test_labels)
    test_labels_binary = lb.transform(test_labels)

    precision = dict()
    recall = dict()
    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(test_labels_binary[:, i], predictions[:, i])

    # Plot Precision-Recall curve for each class
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'turquoise', 'darkgreen', 'red'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-Recall curve of class {0}'
                       ''.format(class_names[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_results(history, test_Y, predicted_labels, predictions):
    """
    Plot the results of the model evaluation.

    Parameters:
    history: History object returned from model.fit()
    test_Y: Array of true labels.
    predicted_labels: Array of predicted labels.
    predictions: Array of predicted probabilities.
    """
    plot_history(history)
    confusion_matrix = plot_confusion_matrix(test_Y, predicted_labels)
    plot_prediction_fractions(confusion_matrix)
    plot_roc_curves(test_Y, predictions)
    plot_precision_recall_curves(test_Y, predictions)
