import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CHARGEMENT DES SPLITS BRUTS
# ============================================================================

print("=" * 80)
print("CHARGEMENT DES SPLITS BRUTS")
print("=" * 80)

df_train = pd.read_csv("data/raw_splits/train.csv")
df_val   = pd.read_csv("data/raw_splits/val.csv")
df_test  = pd.read_csv("data/raw_splits/test.csv")

print(f"✓ Train: {df_train.shape}")
print(f"✓ Val  : {df_val.shape}")
print(f"✓ Test : {df_test.shape}")

# ============================================================================
# FEATURE ENGINEERING (APRÈS SPLIT)
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

for df_ in [df_train, df_val, df_test]:
    df_["start_date"] = pd.to_datetime(df_["start_date"], errors="coerce")
    df_["end_date"]   = pd.to_datetime(df_["end_date"], errors="coerce")

    df_["duration_days"] = (
        df_["end_date"] - df_["start_date"]
    ).dt.days.clip(lower=1)

# ============================================================================
# DÉFINITION DES FEATURES & TARGET
# ============================================================================

features_num = ["goal", "duration_days"]
features_cat = ["subcategory"]

X_train = df_train[features_num + features_cat]
y_train = df_train["target"]

X_val = df_val[features_num + features_cat]
y_val = df_val["target"]

X_test = df_test[features_num + features_cat]
y_test = df_test["target"]

# ============================================================================
# ONE-HOT ENCODING (SANS DATA LEAKAGE)
# ============================================================================

X_train_enc = pd.get_dummies(
    X_train,
    columns=features_cat,
    drop_first=True
)

X_val_enc = pd.get_dummies(
    X_val,
    columns=features_cat,
    drop_first=True
)

X_test_enc = pd.get_dummies(
    X_test,
    columns=features_cat,
    drop_first=True
)

# Alignement strict des colonnes
X_val_enc = X_val_enc.reindex(
    columns=X_train_enc.columns,
    fill_value=0
)

X_test_enc = X_test_enc.reindex(
    columns=X_train_enc.columns,
    fill_value=0
)

print(f"✓ Shape X_train_enc: {X_train_enc.shape}")
print(f"✓ Shape X_val_enc  : {X_val_enc.shape}")
print(f"✓ Shape X_test_enc : {X_test_enc.shape}")

# ============================================================================
# NORMALISATION (k-NN)
# ============================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_enc)
X_val_scaled   = scaler.transform(X_val_enc)
X_test_scaled  = scaler.transform(X_test_enc)

# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_model(name, y_true, y_pred, y_proba=None):
    print(f"\nRÉSULTATS: {name}")
    print("=" * 80)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
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

# ============================================================================
# k-NEAREST NEIGHBORS
# ============================================================================

def check_success_rate(name, y):
    print(f"{name:<12} → success rate = {y.mean():.4f} ({len(y)} échantillons)")

print("\n" + "=" * 80)
print("DISTRIBUTION DES CLASSES")
print("=" * 80)

check_success_rate("TRAIN", y_train)
check_success_rate("VALIDATION", y_val)
check_success_rate("TEST", y_test)

print("\n" + "=" * 80)
print("k-NEAREST NEIGHBORS")
print("=" * 80)

k_values = [3, 5, 7, 11, 15]
best_f1 = 0
best_k = None

for k in k_values:
    print(f"\n→ k = {k}")

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_val_scaled)
    y_proba = knn.predict_proba(X_val_scaled)[:, 1]

    f1 = evaluate_model(f"k-NN (k={k})", y_val, y_pred, y_proba)

    if f1 > best_f1:
        best_f1 = f1
        best_k = k

print(f"\n Meilleur k = {best_k} (F1 = {best_f1:.4f})")


'''
print("\nEntraînement du modèle final...")

knn_final = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_final.fit(X_train_scaled, y_train)

y_test_pred = knn_final.predict(X_test_scaled)
y_test_proba = knn_final.predict_proba(X_test_scaled)[:, 1]

evaluate_model("k-NN FINAL (TEST)", y_test, y_test_pred, y_test_proba)

knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_best.fit(X_train_scaled, y_train)
''''''
# ============================================================================
# 2. ARBRES DE DÉCISION
# ============================================================================

print("\n" + "="*80)
print("2. ARBRES DE DÉCISION")
print("="*80)

print("\n--- GridSearch pour optimiser les hyperparamètres ---")

# Paramètres à tester
param_grid_dt = {
    'max_depth': [5, 10, 15, 20, None],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# GridSearch
dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(
    dt, param_grid_dt, cv=3, scoring='f1', 
    n_jobs=-1, verbose=1
)

print("Entraînement en cours (peut prendre quelques minutes)...")
grid_dt.fit(X_train, y_train)

print(f"\n✓ Meilleurs paramètres: {grid_dt.best_params_}")
print(f"✓ Meilleur score F1 (CV): {grid_dt.best_score_:.4f}")

# Évaluation sur validation
y_pred_dt = grid_dt.predict(X_val)
y_proba_dt = grid_dt.predict_proba(X_val)[:, 1]
dt_metrics = evaluate_model("Arbre de Décision", y_val, y_pred_dt, y_proba_dt)

# Importance des features (top 10)
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': grid_dt.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n--- Top 10 Features les plus importantes ---")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 3. RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("3. RANDOM FOREST")
print("="*80)

print("\n--- GridSearch pour optimiser les hyperparamètres ---")

# Paramètres à tester
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# GridSearch
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_rf = GridSearchCV(
    rf, param_grid_rf, cv=3, scoring='f1',
    n_jobs=-1, verbose=1
)

print("Entraînement en cours (peut prendre plusieurs minutes)...")
grid_rf.fit(X_train, y_train)

print(f"\n✓ Meilleurs paramètres: {grid_rf.best_params_}")
print(f"✓ Meilleur score F1 (CV): {grid_rf.best_score_:.4f}")

# Évaluation sur validation
y_pred_rf = grid_rf.predict(X_val)
y_proba_rf = grid_rf.predict_proba(X_val)[:, 1]
rf_metrics = evaluate_model("Random Forest", y_val, y_pred_rf, y_proba_rf)

# Importance des features (top 10)
feature_importance_rf = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': grid_rf.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n--- Top 10 Features les plus importantes ---")
print(feature_importance_rf.head(10).to_string(index=False))

# ============================================================================
# COMPARAISON DES MODÈLES
# ============================================================================

print("\n" + "="*80)
print("COMPARAISON DES MODÈLES SUR VALIDATION")
print("="*80)

comparison = pd.DataFrame({
    'k-NN': knn_results[best_k],
    'Arbre de Décision': dt_metrics,
    'Random Forest': rf_metrics
}).T

print("\n")
print(comparison.to_string())

# Meilleur modèle
best_model_name = comparison['f1'].idxmax()
print(f"\n🏆 Meilleur modèle (F1-Score): {best_model_name}")
print(f"   F1-Score: {comparison.loc[best_model_name, 'f1']:.4f}")

# ============================================================================
# VISUALISATIONS
# ============================================================================

print("\n" + "="*80)
print("GÉNÉRATION DES VISUALISATIONS")
print("="*80)

# 1. Comparaison des métriques
fig, ax = plt.subplots(figsize=(12, 6))
comparison.plot(kind='bar', ax=ax)
ax.set_title('Comparaison des performances des modèles (Validation)', fontsize=14)
ax.set_xlabel('Modèle', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1)
ax.legend(title='Métriques')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fig_model_comparison.png', dpi=100, bbox_inches='tight')
print("✓ Figure sauvegardée: fig_model_comparison.png")
plt.close()

# 2. Importance des features (Random Forest)
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance_rf.head(15)
ax.barh(range(len(top_features)), top_features['importance'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 15 Features - Random Forest', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('fig_feature_importance_rf.png', dpi=100, bbox_inches='tight')
print("✓ Figure sauvegardée: fig_feature_importance_rf.png")
plt.close()

# 3. Matrice de confusion du meilleur modèle
if best_model_name == 'Random Forest':
    y_pred_best = y_pred_rf
elif best_model_name == 'Arbre de Décision':
    y_pred_best = y_pred_dt
else:
    y_pred_best = knn_best.predict(X_val_scaled)

cm = confusion_matrix(y_val, y_pred_best)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Matrice de Confusion - {best_model_name}', fontsize=14)
ax.set_xlabel('Prédiction', fontsize=12)
ax.set_ylabel('Réalité', fontsize=12)
ax.set_xticklabels(['Échec', 'Succès'])
ax.set_yticklabels(['Échec', 'Succès'])
plt.tight_layout()
plt.savefig('fig_confusion_matrix_best.png', dpi=100, bbox_inches='tight')
print("✓ Figure sauvegardée: fig_confusion_matrix_best.png")
plt.close()

print("\n" + "="*80)
print("✓ ENTRAÎNEMENT ET VALIDATION TERMINÉS")
print("="*80)
print(f"\nProchaine étape: Évaluation sur le jeu de TEST avec le modèle {best_model_name}")'''