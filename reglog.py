import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

print("=" * 80)
print("CHARGEMENT ET PRÉPARATION DES DONNÉES")
print("=" * 80)

df_clean = pd.read_csv("data/ks-projects-clean.csv", encoding="ISO-8859-1")

# Conversion des dates
df_clean["start_date"] = pd.to_datetime(df_clean["start_date"], errors="coerce")
df_clean["end_date"] = pd.to_datetime(df_clean["end_date"], errors="coerce")

# Suppression des lignes avec dates invalides
df_clean = df_clean.dropna(subset=["start_date", "end_date"])

print(f"✓ Données chargées: {df_clean.shape}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Durée de la campagne
df_clean["duration_days"] = (df_clean["end_date"] - df_clean["start_date"]).dt.days
df_clean["duration_days"] = df_clean["duration_days"].clip(lower=1)

# Variable cible
df_clean["target"] = (df_clean["state"] == "successful").astype(int)

# Regroupement des catégories rares
min_count = 20
df_clean["subcategory"] = df_clean["subcategory"].where(
    df_clean["subcategory"].map(df_clean["subcategory"].value_counts()) >= min_count,
    "Other"
)

# Clé de stratification multi-critères
df_clean["stratify_key"] = (
        df_clean["target"].astype(str) + "_" + df_clean["subcategory"]
)

print("✓ Feature engineering terminé")

# PRÉPARATION DES FEATURES
# ============================================================================

# features_num = ["goal", "age", "pledged", "backers", "duration_days"]
features_num = ["goal", "age", "duration_days"]
features_cat1 = ["category", "subcategory", "country", "sex"]
features_cat2 = ["category", "country", "sex"]
features_cat3 = ["subcategory", "country", "sex"]
features_cat4 = ["category", "country"]
features_cat5 = ["subcategory", "country"]

y = df_clean["target"]
strat = df_clean["stratify_key"]

def config(features_cat) :
    x = df_clean[features_num + features_cat]

    # DÉCOUPAGE STRATIFIÉ 60 / 20 / 20
    # ============================================================================

    # print("\n" + "=" * 80)
    # print("DÉCOUPAGE STRATIFIÉ DES DONNÉES")
    # print("=" * 80)

    X_temp, X_test, y_temp, y_test, strat_temp, strat_test = train_test_split(
        x, y, strat,
        test_size=0.20,
        random_state=42,
        stratify=strat
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=strat_temp
    )

    # print(f"✓ Train: {len(X_train)}")
    # print(f"✓ Validation: {len(X_val)}")
    # print(f"✓ Test: {len(X_test)}")
    return X_temp, X_test, y_temp, y_test, strat_temp, strat_test, X_train, X_val, y_train, y_val


# ÉVALUATION
# ============================================================================

def evaluate_model(name, y_true, y_pred, y_proba=None):
    print(f"\n{'=' * 80}")
    print(f"RÉSULTATS: {name}")
    print(f"{'=' * 80}")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"AUC-ROC : {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nMatrice de confusion:")
    print(cm)

    return f1


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

    #     Pour la régression logistique :
    # la normalisation min–max n’est pas nécessaire
    # le centrage–réduction (standardisation) est fortement recommandé


    # centrage + réduction = numérique
    # One-Hot Encoding = catégorielle
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


def evaluer(name, features_cat) :

    X_temp, X_test, y_temp, y_test, strat_temp, strat_test, X_train, X_val, y_train, y_val = config(features_cat)

    logreg_pipe, logreg_val_metrics = train_logistic_regression(
        X_train, y_train,
        X_val, y_val,
        features_num,
        features_cat
    )

    # Prédictions test
    y_test_pred = logreg_pipe.predict(X_test)
    y_test_proba = logreg_pipe.predict_proba(X_test)[:, 1]

    # Évaluation unifiée
    evaluate_model(
        name,
        y_test,
        y_test_pred,
        y_test_proba
    )
    print("x" * 80)
    print()

evaluer("Config1 : Category, Subcategory, Country, Sex", features_cat1)
evaluer("Config2 : Category, Country, Sex", features_cat2)
evaluer("Config3 : Subcategory, Country, Sex", features_cat3)
evaluer("Config4 : Category, Country", features_cat4)
evaluer("Config5 : Subcategory, Country", features_cat5)