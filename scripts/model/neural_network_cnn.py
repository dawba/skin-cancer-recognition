import time
from keras.src.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import pandas as pd


class NeuralNetworkCNN:
    def __init__(self, labels, model_builder, validation_split=0.2, batch_size=32, epochs=10, images=None,
                 metadata=None):
        if isinstance(labels, pd.Series):
            labels = labels.values

        self.labels = labels
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.model_builder = model_builder
        self.epochs = epochs
        self.images = images
        self.metadata = metadata

    def train_model(self, train_images, train_metadata, train_labels, val_images, val_metadata, val_labels,
                    hyperparameters):
        class_weights = compute_class_weight('balanced', classes=np.unique(self.labels), y=self.labels)
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}

        model = self.model_builder.build_model(
            image_shape=self.images.shape[1:],
            metadata_shape=self.metadata.shape[1],
            num_classes=len(np.unique(self.labels)),
            M=hyperparameters['M'], N=hyperparameters['N'], K=hyperparameters['K'],
            learning_rate=hyperparameters['learning_rate'],
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)
        callbacks = [early_stopping, reduce_lr]

        history = History()

        print("Starting training...")
        history = model.fit(
            [train_images, train_metadata],
            train_labels,
            epochs=hyperparameters['epochs'],
            validation_data=([val_images, val_metadata], val_labels),
            class_weight=class_weights,
            batch_size=hyperparameters['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        print("Network trained!")
        return model, history

    def grid_search(self, param_grid):
        best_score = float('-inf')
        best_params = None
        best_score = 0

        # Split the data once for validation
        train_images, val_images, train_metadata, val_metadata, train_labels, val_labels = train_test_split(
            self.images, self.metadata, self.labels, test_size=self.validation_split, stratify=self.labels)

        for params in param_grid:
            start_time = time.time()
            print(f"Training with parameters: {params}")
            model, history = self.train_model(train_images, train_metadata, train_labels, val_images, val_metadata,
                                              val_labels, params)
            val_acc = max(history.history['val_accuracy'])
            end_time = time.time()
            print(f"Time elapsed for this training session: {self.__format_time(end_time - start_time)}")
            print(f"Validation accuracy: {val_acc}")

            if val_acc > best_score:
                best_score = val_acc
                best_params = params
                best_model = model

                model_file = "cnn_" + str(best_score) + ".pkl"
                param_file = "cnn_" + str(best_score) + ".txt"
                joblib.dump(best_model, model_file)
                with open(param_file, "w") as f:
                    f.write(str(best_params))

        print("Best params found during grid search:", best_params)
        print("Training final model on the entire dataset...")

        # Training the final model on the entire dataset with the best found hyperparameters
        model, history = self.train_model(self.images, self.metadata, self.labels, self.images, self.metadata,
                                          self.labels, best_params)

        model_file = "cnn_final_" + str(best_score) + ".pkl"
        joblib.dump(model, model_file)

        with open("params_final.txt", "w") as f:
            f.write(str(best_params))

        return model, best_params, history

    def __format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f'{int(hrs):02}:{int(mins):02}:{int(secs):02}'
