import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize


def build_datasets(X, y, test_size, val_size, random_state):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    unique, counts = np.unique(y_encoded, return_counts=True)
    singleton = unique[counts == 1]
    multi = unique[counts > 1]

    mask_single = np.isin(y_encoded, singleton)
    mask_multi = ~mask_single

    X_single, y_single = X[mask_single], y_encoded[mask_single]
    X_multi, y_multi = X[mask_multi], y_encoded[mask_multi]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_multi, y_multi,
        test_size=test_size,
        random_state=random_state,
        stratify=y_multi
    )

    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval
    )

    X_train = np.vstack([X_train_main, X_single])
    y_train = np.concatenate([y_train_main, y_single])

    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        le, n_classes
    )