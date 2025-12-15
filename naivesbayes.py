import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_naive_bayes(X_train, y_train, X_val=None, y_val=None,
                      numeric_cols=None, categorical_cols=None,
                      save_path='artifacts/best_nb.pkl'):
    """
    Entraîne un modèle Naive Bayes avec pipeline.
    Évalue sur validation si fourni et sauvegarde le modèle.
    """
    # Préprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])

    # Entraînement
    pipe.fit(X_train, y_train)

    # Sauvegarde
    joblib.dump(pipe, save_path)
    print(f"[train_naive_bayes] Modèle sauvegardé dans : {save_path}")

    # Évaluation sur validation
    metrics_val = None
    if X_val is not None and y_val is not None:
        y_pred = pipe.predict(X_val)
        metrics_val = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred),
            'classification_report': classification_report(y_val, y_pred)
        }
        print("[train_naive_bayes] Évaluation sur validation :")
        print(f"  Accuracy: {metrics_val['accuracy']:.4f}")
        print(f"  F1-score: {metrics_val['f1_score']:.4f}")

    return pipe, metrics_val
