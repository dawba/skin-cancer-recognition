from scripts.utils.preprocess_data import lesion_type_dict
import numpy as np

def split_X(images, metadata, validation_split=0.2):
    """
    Split the dataset into training and validation sets.

    Parameters:
    - images: NumPy array of images.
    - metadata: DataFrame containing metadata.
    - validation_split: Fraction of the data to use for validation (default is 0.2).

    Returns:
    - (train_images, train_metadata): Training set of images and metadata.
    - (val_images, val_metadata): Validation set of images and metadata.
    """
    data_size = len(images)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    val_size = int(data_size * validation_split)
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_images = images[train_indices]
    val_images = images[val_indices]

    train_metadata = metadata.iloc[train_indices].reset_index(drop=True)
    val_metadata = metadata.iloc[val_indices].reset_index(drop=True)

    return (train_images, train_metadata), (val_images, val_metadata)


def extract_labels(metadata):
    """
    Extract labels from metadata based on a predefined dictionary.

    Parameters:
    - metadata: DataFrame containing metadata with a 'dx' column.

    Returns:
    - labels: Series of labels mapped from the 'dx' column.
    """
    labels = metadata['dx'].map({v['code']: k for k, v in lesion_type_dict.items()})
    return labels


def split_data(images, metadata, validation_split=0.2):
    """
    Split the data into training and validation sets, extracting labels and dropping the 'dx' column.

    Parameters:
    - images: NumPy array of images.
    - metadata: DataFrame containing metadata with a 'dx' column.
    - validation_split: Fraction of the data to use for validation (default is 0.2).

    Returns:
    - (train_images, train_metadata): Training set of images and metadata without 'dx' column.
    - (val_images, val_metadata): Validation set of images and metadata without 'dx' column.
    - train_labels: Training labels extracted from 'dx' column.
    - val_labels: Validation labels extracted from 'dx' column.
    """
    print('Splitting data...')

    (train_images, train_metadata), (val_images, val_metadata) = split_X(images, metadata, validation_split)

    train_labels = extract_labels(train_metadata)
    val_labels = extract_labels(val_metadata)

    train_metadata = train_metadata.drop(columns=['dx'])
    val_metadata = val_metadata.drop(columns=['dx'])

    print('Split!')
    return (train_images, train_metadata), (val_images, val_metadata), train_labels, val_labels
