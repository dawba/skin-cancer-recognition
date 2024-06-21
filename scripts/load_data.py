from PIL import Image
import numpy as np
import pandas as pd
import os
import shutil

folder1 = '../../data/HAM10000/HAM10000_images_part_1'
folder2 = '../../data/HAM10000/HAM10000_images_part_2'
destination_folder = '../../data/HAM10000/HAM10000_images'
metadata_file = '../../data/HAM10000/HAM10000_metadata.csv'
augmented_folder = '../../data/HAM10000/HAM10000_images_augmented'

''' 
    
    Generate a list of image IDs for each dx value in the metadata.
    Only used, when number of samples is required for each class.
    However, we found that it is better to train this model on imbalanced data.

'''
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


def load_data(folder1, folder2, destination_folder, metadata_file, image_dimension=96):
    """
    Load data from two folders of images and a metadata CSV file.

    Parameters:
    - folder1: Path to the first folder of images.
    - folder2: Path to the second folder of images.
    - destination_folder: Path to the destination folder where merged images will be stored.
    - metadata_file: Path to the metadata CSV file containing image labels.
    - num_samples: Number of samples per class to include in the dataset (optional).

    Returns:
    - images: NumPy array of resized images.
    - labels: NumPy array of corresponding labels.
    - metadata: DataFrame containing metadata excluding the label column.
    """
    print(f'Loading data...')

    # Create destination folder
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    os.makedirs(destination_folder)

    # Load metadata
    metadata = pd.read_csv(metadata_file)
    metadata = metadata.drop_duplicates()

    # Check if num_samples is specified
    # if num_samples is not None:
    #     # Get unique dx values for sampling
    #     image_id_dict = generate_image_id_list(metadata, num_samples=num_samples)
    #     selected_image_ids = []
    #     for image_ids in image_id_dict.values():
    #         selected_image_ids.extend(image_ids)
    #     selected_image_ids = set(selected_image_ids)
    #
    #     # Filter metadata to include only selected image IDs
    #     metadata = metadata[metadata['image_id'].isin(selected_image_ids)]

    # Merge two folders from the original format
    for folder in [folder1, folder2]:
        for filename in os.listdir(folder):
            source = os.path.join(folder, filename)
            destination = os.path.join(destination_folder, filename)
            shutil.copy(source, destination)

    # Load images and compress them for efficient format
    images = []
    for index, row in metadata.iterrows():
        img_path = os.path.join(destination_folder, row['image_id'] + '.jpg')
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((image_dimension, image_dimension))
            images.append(np.array(image))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    images = np.array(images)

    print('Loaded!')
    return images, metadata