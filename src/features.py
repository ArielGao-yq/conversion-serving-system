from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
def build_preprocessor(numeric_features, categorical_features):
    preprocessor=ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ],
        sparse_threshold=0
    )
    return preprocessor