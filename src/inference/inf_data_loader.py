import ast
import numpy as np
import pandas as pd


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    df["body_embeddings"] = df["body_embeddings"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
    )

    new_indices = df.index.tolist()

    return df, new_indices