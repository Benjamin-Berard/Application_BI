import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_logistic_regression(X_train, y_train, X_val=None, y_val=None,
                              numeric_cols=None, categorical_cols=None,
                              save_path='artifacts/best_logreg.pkl'):
    """
    Entraîne un modèle de régression logistique avec pipeline,
    évalue sur X_val si fourni, et sauvegarde le modèle.

    Args:
        X_train, y_train : données d'entraînement
        X_val, y_val : données de validation (optionnel)
        numeric_cols : liste des colonnes numériques
        categorical_cols : liste des colonnes catégorielles
        save_path : chemin pour sauvegarder le modèle
    Returns:
        model : pipeline entraîné
        metrics_val : dictionnaire des métriques sur validation si X_val fourni
    """
    # --- Préprocessing ---
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # --- Pipeline complet ---
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # --- Entraînement ---
    pipe.fit(X_train, y_train)

    # --- Sauvegarde du modèle ---
    joblib.dump(pipe, save_path)
    print(f"[train_logistic_regression] Modèle sauvegardé dans : {save_path}")

    # --- Évaluation sur validation ---
    metrics_val = None
    if X_val is not None and y_val is not None:
        y_pred = pipe.predict(X_val)
        metrics_val = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred),
            'classification_report': classification_report(y_val, y_pred)
        }
        print("[train_logistic_regression] Évaluation sur validation :")
        print(f"  Accuracy: {metrics_val['accuracy']:.4f}")
        print(f"  F1-score: {metrics_val['f1_score']:.4f}")

    return pipe, metrics_val
