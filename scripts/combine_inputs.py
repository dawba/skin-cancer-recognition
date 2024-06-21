import numpy as np

def combine_inputs(images, metadata):
    """
    Combine flattened image data with transformed metadata.

    Parameters:
    - images: NumPy array of images.
    - metadata: NumPy array of transformed metadata.

    Returns:
    - combined_input: Combined input array.
    """
    # Flatten image data
    flattened_images = images.reshape((images.shape[0], -1))
    # Combine with metadata
    combined_input = np.concatenate([flattened_images, metadata], axis=1)

    print('Inputs combined!')
    return combined_input
