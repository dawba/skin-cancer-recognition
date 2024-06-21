from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.image import resize
import os
import shutil
import numpy as np
import pandas as pd

lesion_type_dict = {
    0: {'code': 'nv', 'name': 'Melanocytic nevi'},
    1: {'code': 'mel', 'name': 'Melanoma'},
    2: {'code': 'bkl', 'name': 'Benign keratosis-like lesions'},
    3: {'code': 'bcc', 'name': 'Basal cell carcinoma'},
    4: {'code': 'akiec', 'name': 'Actinic keratoses'},
    5: {'code': 'vasc', 'name': 'Vascular lesions'},
    6: {'code': 'df', 'name': 'Dermatofibroma'}
}

def augment_images(images, metadata, augmented_folder, batch_size=2, target_size=(28, 28)):
    """
    Augment images by applying various transformations and save them to a folder.

    Parameters:
    - images: NumPy array of images to be augmented.
    - metadata: DataFrame containing metadata of the images.
    - batch_size: Number of augmentations to be generated per image.
    - target_size: Tuple indicating the desired size of the augmented images.

    Returns:
    - all_images: NumPy array containing original and augmented images.
    - all_metadata: DataFrame containing metadata for original and augmented images.
    """

    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: np.rot90(x, k=np.random.choice([1, 2, 3])),
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=(1, 1.2),
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create the folder for augmented images if it does not exist
    if os.path.exists(augmented_folder):
        shutil.rmtree(augmented_folder)
    os.makedirs(augmented_folder)

    # DataFrames and lists to store augmented images and metadata
    augmented_metadata = pd.DataFrame()
    augmented_images = []

    # Augment each image
    for i in range(len(images)):
        image = images[i]
        image = img_to_array(image).reshape((1,) + image.shape)

        # Generate augmented images
        aug_iter = datagen.flow(image, batch_size=batch_size)
        aug_images = [next(aug_iter)[0] for _ in range(batch_size)]

        for j, aug_image in enumerate(aug_images):
            aug_image = aug_image.astype(np.uint8)

            # Resize the image to the target size
            aug_image = resize(aug_image, target_size)
            aug_image = aug_image.numpy().astype(np.uint8)

            img = array_to_img(aug_image)

            # Save augmented image
            aug_image_id = f"{metadata.iloc[i]['image_id']}_aug_{j}"
            img_path = os.path.join(augmented_folder, f"{aug_image_id}.jpg")
            img.save(img_path)

            # Append metadata for augmented image
            new_metadata = pd.DataFrame({
                'image_id': [aug_image_id],
                'lesion_id': [metadata.iloc[i]['lesion_id']],
                'dx': [metadata.iloc[i]['dx']],
                'dx_type': [metadata.iloc[i]['dx_type']],
                'age': [metadata.iloc[i]['age']],
                'sex': [metadata.iloc[i]['sex']],
                'localization': [metadata.iloc[i]['localization']]
            })
            augmented_metadata = pd.concat([augmented_metadata, new_metadata], ignore_index=True)

            augmented_images.append(aug_image)

    # Combine original and augmented data
    resized_images = np.array([resize(image, target_size).numpy().astype(np.uint8) for image in images])
    all_images = np.concatenate((resized_images, np.array(augmented_images)), axis=0)
    all_metadata = pd.concat([metadata, augmented_metadata], ignore_index=True)

    print('Augmented!')
    return all_images, all_metadata

def sanitize_data(metadata):
    """
    Clean metadata by filling missing values and drop redundant columns.

    Parameters:
    - metadata: DataFrame containing metadata of the images.
    """
    print('Sanitizing data...')

    # Drop redundant columns from learning standpoint
    metadata = metadata.drop(columns=['lesion_id', 'image_id'], errors='ignore')

    # Fill missing age with mean age
    metadata['age'] = metadata['age'].fillna(metadata['age'].mean())

    categorical_columns = ['dx_type', 'sex', 'localization']

    for col in categorical_columns:
        # Calculate the value counts and normalize to get probabilities
        value_counts = metadata[col].value_counts(normalize=True)

        # Generate random choices based on the probability distribution
        missing_indices = metadata[metadata[col].isna()].index
        random_choices = np.random.choice(value_counts.index, size=len(missing_indices), p=value_counts.values)

        # Fill missing values with the random choices
        metadata.loc[missing_indices, col] = random_choices

    print('Sanitized!')
    return metadata


def preprocess_data(images, metadata, augmented_folder=None, augmentation=True, target_size=(28, 28)):
    """
    Arranges preprocess steps into a pipeline.

    Parameters:
    - images: NumPy array of images to be preprocessed.
    - metadata: DataFrame containing metadata of the images.
    - augmented_folder: Path to the folder for augmented images.
    - augmentation: Boolean flag to indicate if augmentation should be performed.
    - target_size: Tuple indicating the desired size of the images.

    Returns:
    - images: NumPy array of preprocessed images.
    - metadata: DataFrame containing cleaned metadata.
    """
    print('Preprocessing data...')

    if augmentation and augmented_folder:
        images, metadata = augment_images(images, metadata, augmented_folder, target_size=target_size)
    else:
        # Resize images without augmentation
        resized_images = np.array([resize(image, target_size).numpy().astype(np.uint8) for image in images])
        images = resized_images

    metadata = sanitize_data(metadata)

    print('Preprocessed!')
    return images, metadata

