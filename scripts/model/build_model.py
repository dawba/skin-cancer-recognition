from keras import Model
from tensorflow.python.ops.init_ops_v2 import he_normal
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ReLU, concatenate, Reshape, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np


class ModelBuilder:
    def __init__(self):
        pass

    def build_model(self, image_shape, metadata_shape, num_classes, M=2, N=2, K=2, learning_rate=0.001):
        """
        Build a convolutional neural network model with combined image and metadata input.

        Parameters:
        - image_shape: Shape of the input images.
        - metadata_shape: Shape of the metadata input.
        - num_classes: Number of output classes.
        - M: Number of blocks of convolutional layers for image data.
        - N: Number of convolutional layers per block.
        - K: Number of fully connected layers for metadata.
        - learning_rate: Learning rate for the Adam optimizer.

        Returns:
        - model: Compiled Keras model.
        """

        combined_input = Input(shape=(np.prod(image_shape) + metadata_shape,), name='combined_input')

        # Split combined input into image and metadata parts
        image_input = combined_input[:, :np.prod(image_shape)]
        metadata_input = combined_input[:, np.prod(image_shape):]

        # Reshape the image input
        image_input = Reshape(image_shape)(image_input)

        x = image_input
        # Convolutional layers for image input
        for i in range(M):
            for _ in range(N):
                x = Conv2D(32 * (2 ** i), (3, 3), padding='same', kernel_initializer=he_normal())(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)  # reduce image spatial dimensions

        x = Flatten()(x)

        # Metadata input
        y = metadata_input
        for _ in range(K):
            y = Dense(128)(y)
            y = BatchNormalization()(y)
            y = ReLU()(y)

        # Concatenate both paths
        combined = concatenate([x, y])

        # Final fully connected layers
        z = Dense(1024, kernel_initializer=he_normal())(
            combined)
        z = BatchNormalization()(z)
        z = ReLU()(z)

        z = Dense(num_classes, activation='softmax', kernel_initializer=he_normal())(z)

        model = Model(inputs=[image_input, metadata_input], outputs=z)

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
