from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def encode_metadata(metadata, fit=True, encoder=None):
    """
    Transform the metadata using standard scaling for numerical features
    and one-hot encoding for categorical features.

    Parameters:
    - metadata: DataFrame containing metadata.
    - fit: Boolean indicating if the encoder should be fitted (True for training data).
    - encoder: Pre-fitted encoder (used if fit is False).

    Returns:
    - metadata_transformed: Transformed metadata as a NumPy array.
    - encoder: Fitted encoder (if fit is True).
    """

    if fit:
        categorical_features = ['dx_type', 'sex', 'localization']
        numeric_features = ['age']

        dx_type_categories = ['histo', 'consensus', 'confocal', 'follow_up']
        sex_categories = ['male', 'female', 'unknown']
        localization_categories = [
            'scalp', 'ear', 'face', 'back', 'trunk', 'chest', 'upper extremity',
            'abdomen', 'unknown', 'lower extremity', 'genital', 'neck', 'hand',
            'foot', 'acral'
        ]

        encoder = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(categories=[dx_type_categories, sex_categories, localization_categories],
                                      sparse_output=False), categorical_features)
            ])
        metadata_transformed = encoder.fit_transform(metadata)
    else:
        metadata_transformed = encoder.transform(metadata)

    print('Metadata transformed!')
    return metadata_transformed, encoder
