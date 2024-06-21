import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer


def load_metadata(metadata_dir):
    metadata = pd.read_csv(metadata_dir)
    return metadata


def generate_image_id_list(metadata, num_samples=100):
    # Filter metadata to get unique dx values
    unique_dx = metadata['dx'].unique()

    # Initialize an empty dictionary to store image IDs for each dx
    image_id_dict = {}

    # Iterate through unique dx values
    for dx_value in unique_dx:
        # Filter metadata for current dx
        dx_metadata = metadata[metadata['dx'] == dx_value]

        # Get image IDs for the current dx
        image_ids = dx_metadata['image_id'].unique()

        # Ensure we have at least num_samples image IDs for this dx
        if len(image_ids) >= num_samples:
            # Randomly select num_samples unique image_id for this dx
            selected_image_ids = np.random.choice(image_ids, num_samples, replace=False)
        else:
            # If there are fewer than num_samples, select all of them
            selected_image_ids = image_ids

        # Store the selected image IDs for this dx
        image_id_dict[dx_value] = selected_image_ids

    return image_id_dict


def load_images_for_augmentation(metadata, image_dir, image_id_dict, target_size=(220, 220)):
    images = []
    labels = []

    for key in image_id_dict:
        for value in image_id_dict[key]:
            image_id = value
            label = key
            image_path = os.path.join(image_dir, image_id + '.jpg')
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)

    images = np.array(images)
    images = images / 255.0  # Normalize pixel values
    return images, labels


def augment_images(images, labels):
    # Initialize ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Make sure labels are in correct format
    labels = np.array(labels)

    # Encode the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        img = np.expand_dims(img, 0)  # Expand dimensions to match ImageDataGenerator input format
        num_augmented_images = 0  # Track the number of augmented images generated for each original image
        for batch in datagen.flow(img, batch_size=1):
            aug_img = batch[0].astype('float32')
            aug_img = (aug_img * 255).astype('uint8')  # Re-scale to original range
            augmented_images.append(aug_img)
            augmented_labels.append(label)
            num_augmented_images += 1
            if num_augmented_images >= 2:  # Augment each image 2 times
                break

    augmented_images = np.array(augmented_images).squeeze()
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels


def load_balanced_data_pipeline(metadata_dir, image_dir):
    print("Loading balanced data...")
    metadata = load_metadata(metadata_dir)
    print("Metadata loaded.")

    print("Generating image ID list...")
    image_id_dict = generate_image_id_list(metadata)
    print("Image ID list generated.")

    print("Loading images for augmentation...")
    images, labels = load_images_for_augmentation(metadata, image_dir, image_id_dict)

    print("Augmenting images...")
    augmented_images, augmented_labels = augment_images(images, labels)
    print("Images augmented.")


    # Convert labels to appropriate data type
    augmented_labels = np.array(augmented_labels, dtype=np.float32)

    print("Data loading complete.")
    return augmented_images, augmented_labels
