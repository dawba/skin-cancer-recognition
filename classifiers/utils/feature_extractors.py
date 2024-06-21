import cv2
import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import graycomatrix, graycoprops
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Custom transformer to extract color histogram features
class ColorHistogramTransformer(BaseEstimator, TransformerMixin):
    # Function to extract color histogram features
    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        image = image.astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extract_color_histogram(image) for image in X])

# Custom transformer to extract Hu Moments features
class HuMomentsTransformer(BaseEstimator, TransformerMixin):
    # Function to extract Hu Moments features
    def extract_hu_moments(self, image):
        image = image.astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments)
        return hu_moments.flatten()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extract_hu_moments(image) for image in X])

# Custom transformer to extract Haralick texture features
class HaralickFeaturesTransformer(BaseEstimator, TransformerMixin):
    # Function to extract Haralick texture features
    def extract_haralick_features(self, image):
        image = image.astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
        haralick_features = []
        for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'):
            haralick_features.append(graycoprops(glcm, prop).mean())
        return np.array(haralick_features)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extract_haralick_features(image) for image in X])

# Custom Metadata Transformer
class MetadataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_features = ['dx_type', 'sex', 'localization']
        self.numeric_features = ['age']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(sparse_output=False), self.categorical_features)
            ])

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)