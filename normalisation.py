# ============================================================
# 5. PRÉTRAITEMENT – NORMALISATION DES VARIABLES NUMÉRIQUES
# ============================================================

# Améliore la convergence de la régression logistique
# Indispensable pour :
# kNN
# SVM
# méthodes à base de gradient
# Rend les coefficients comparables entre variables

from sklearn.preprocessing import StandardScaler

def normaliser(df_clean):

    # Sélection des variables numériques à normaliser
    numeric_cols = [
        'age',
        'goal',
        'pledged',
        'backers',
        'duration_days'
    ]

    print("\nNORMALISATION DES VARIABLES NUMÉRIQUES")
    print("Variables concernées :", numeric_cols)

    # Initialisation du scaler
    scaler = StandardScaler()

    # Ajustement + transformation
    df_clean_scaled = df_clean.copy()
    df_clean_scaled[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    print("\nStatistiques APRÈS normalisation (doivent être centrées/réduites) :")
    print(df_clean_scaled[numeric_cols].describe())

    return df_clean_scaled
