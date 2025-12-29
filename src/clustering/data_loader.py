import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_embeddings_and_texts(csv_path: str):
    df = pd.read_csv(csv_path)

    df["body_embeddings"] = df["body_embeddings"].apply(
        lambda x: np.array(ast.literal_eval(x))
    )

    texts = df["Cleaned_body"].fillna("").tolist()
    X = np.vstack(df["body_embeddings"])
    indices = np.arange(len(df))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, texts, indices