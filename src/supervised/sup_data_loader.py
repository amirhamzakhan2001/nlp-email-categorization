import ast
import numpy as np
import pandas as pd


def parse_embedding(x):
    return np.array(ast.literal_eval(str(x)), dtype=np.float32)


def load_csv_embeddings(csv_path: str):
    df = pd.read_csv(csv_path)

    embeddings = []
    labels = []
    failed = 0

    for _, row in df.iterrows():
        try:
            embeddings.append(parse_embedding(row["body_embeddings"]))
            labels.append(row["cluster_id"])
        except Exception:
            failed += 1

    X = np.vstack(embeddings)
    y = np.array(labels)

    return X, y, df