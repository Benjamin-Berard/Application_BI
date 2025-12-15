import json

import pandas as pd
from sklearn.model_selection import train_test_split

def make_splits(df: pd.DataFrame, target_col: str = 'success', seed: int = 42, test_size: float = 0.2, val_size: float = 0.2):
    """
    Crée et sauvegarde des splits train/val/test stratifiés.

    Arguments
    ---------
    df : pd.DataFrame
        Dataframe complet (avec la colonne target_col).
    target_col : str
        Nom de la colonne cible (binaire 0/1).
    seed : int
        Seed pour la reproductibilité.
    test_size : float
        Fraction finale du dataset allouée au jeu test.
    val_size : float
        Fraction finale du dataset allouée au jeu validation.

    Retour
    ------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    # Séparer X (features) et y (target)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ici on veut deux découpes successives : d'abord retirer test_size, puis séparer validation/test sur le reste
    # Option: si test_size + val_size >= 1.0 -> erreur
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size doit être < 1.0")

    # Première séparation : train vs temp (val+test)
    remaining = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=remaining, stratify=y, random_state=seed
    )

    # Deuxième séparation : validation vs test (proportionnellement sur X_temp)
    # fraction de validation par rapport à X_temp : val_size / (val_size + test_size)
    val_frac_of_temp = val_size / remaining
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_frac_of_temp), stratify=y_temp, random_state=seed
    )

    # Sauvegarde des indices pour reproductibilité
    splits = {
        'train_idx': X_train.index.astype(int).tolist(),
        'val_idx': X_val.index.astype(int).tolist(),
        'test_idx': X_test.index.astype(int).tolist()
    }
    with open('artifacts/splits.json', 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    print(f"[make_splits] Splits créés (train={len(X_train)}, val={len(X_val)}, test={len(X_test)})")
    return X_train, X_val, X_test, y_train, y_val, y_test
