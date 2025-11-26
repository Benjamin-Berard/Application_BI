

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('data/ks-projects.csv', encoding='ISO-8859-1')
print(f"il y a : {df.shape[0]} lignes, et  {df.shape[1]} colonnes")
print(df.head())


print("ANALYSE DESCRIPTIVE")

print("\n Types de données")
print(df.dtypes)

print("\n Valeurs manquantes")
valeurs_manquantes = df.isnull().sum()
print(valeurs_manquantes)


print(" ANALYSE DÉTAILLÉE PAR ATTRIBUT")

print("\n Variable CIBLE: state")
print(df['state'].value_counts())


print("\n Attribut: category" )
print(df['category'].value_counts().head(10))

print("\n Attribut: subcategory ")
print(f"Nombre de sous-catégories: {df['subcategory'].nunique()}")
print(df['subcategory'].value_counts().head(10))

print("\n Attribut: country ")
print(df['country'].value_counts())

print("\n Attribut: sex ")
print(df['sex'].value_counts())
print(f"Valeurs manquantes: {df['sex'].isnull().sum()}")

print("\n Attribut: age ")
print(df['age'].describe())
print(f"Valeurs manquantes: {df['age'].isnull().sum()}")
print(f"Valeurs négatives: {(df['age'] < 0).sum()}")
print(f"Valeurs > 100: {(df['age'] > 100).sum()}")

print("\n Attribut: currency ")
print(df['currency'].value_counts().head(10))

print("\n Attributs: goal et pledged ")
print(f"Goal - Min: {df['goal'].min()}, Max: {df['goal'].max()}, Médiane: {df['goal'].median()}")
print(f"Pledged - Min: {df['pledged'].min()}, Max: {df['pledged'].max()}, Médiane: {df['pledged'].median()}")
print(f"Goal = 0: {(df['goal'] == 0).sum()}")
print(f"Pledged = 0: {(df['pledged'] == 0).sum()}")

# Backers
print("\n Attribut: backers ")
print(df['backers'].describe())
print(f"Backers = 0: {(df['backers'] == 0).sum()}")
print(f"Valeurs manquantes: {df['backers'].isnull().sum()}")

# Dates
print("\n Attributs: start_date et end_date ")
print(f"Type de start_date: {df['start_date'].dtype}")
print(f"Type de end_date: {df['end_date'].dtype}")
print(f"Dates manquantes: {(df['start_date'].isnull() | df['end_date'].isnull()).sum()}")

# ============================================================================
# 2.4 DÉTECTION DES PROBLÈMES
# ============================================================================
print("\n" + "="*80)
print("[2.4] DÉTECTION DES PROBLÈMES")
print("="*80)


colonnes_avec_manquantes = valeurs_manquantes[valeurs_manquantes > 0]

if len(colonnes_avec_manquantes) > 0:
    print(f"Il y a {len(colonnes_avec_manquantes)} attributs avec des valeurs manquantes :")
    for col, nb in colonnes_avec_manquantes.items():
        print(f" - {col} : {nb} valeurs manquantes")

    
if (df['goal'] == 0).sum() > 0:
    print(f"{(df['goal'] == 0).sum()} projets avec goal = 0 (incohérent)")

if df['start_date'].dtype == 'object':
    print("Dates au format texte,on doit convertir en datetime")


print("\nProblèmes identifiés:")


# ============================================================================
# 2.5 NETTOYAGE DES DONNÉES
# ============================================================================

print("NETTOYAGE DES DONNÉES")

df_clean = df.copy()
print(f"Taille initiale: {df_clean.shape}")

# Supprimer les attributs inutiles (id, name)
print("\n Suppression des attributs 'id' et 'name'")
df_clean = df_clean.drop(['id', 'name'], axis=1)

# Filtrer les états pertinents (successful vs failed)
print("\n Filtrage des états pertinents")
print(f"  États avant filtrage: {df_clean['state'].unique()}")
# On garde seulement 'successful' et 'failed' pour un problème de classification binaire
etats_pertinents = ['successful', 'failed']
df_clean = df_clean[df_clean['state'].isin(etats_pertinents)]
print(f"  Lignes après filtrage: {df_clean.shape[0]}")
print(f"  Distribution: {df_clean['state'].value_counts()}")



# Traiter les valeurs manquantes de 'sex' et 'age'
print("\n Traitement des valeurs manquantes")
# Option: imputation par le mode pour sex, médiane pour age
df_clean['sex'].fillna(df_clean['sex'].mode()[0], inplace=True)


# Conversion des dates
print("\n Conversion des dates")
df_clean['start_date'] = pd.to_datetime(df_clean['start_date'])
df_clean['end_date'] = pd.to_datetime(df_clean['end_date'])

print(f"\nTaille finale après nettoyage: {df_clean.shape}")
print(f"Valeurs manquantes restantes:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# ============================================================================
# 2.6 VISUALISATIONS
# ============================================================================

print("GRAPHE")


# Variable cible
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_clean['state'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Distribution de la variable cible (state)')
axes[0].set_xlabel('État')
axes[0].set_ylabel('Nombre de projets')
axes[0].tick_params(axis='x', rotation=45)

df_clean['state'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Proportions de succès/échec')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('fig1_target_distribution.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée: fig1_target_distribution.png")
plt.close()

# Distributions des variables numériques
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Goal (log scale)
axes[0].hist(np.log10(df_clean['goal'] + 1), bins=50, edgecolor='black')
axes[0].set_title('Distribution de goal (log10)')
axes[0].set_xlabel('log10(goal)')
axes[0].set_ylabel('Fréquence')

# Pledged (log scale)
axes[1].hist(np.log10(df_clean['pledged'] + 1), bins=50, edgecolor='black')
axes[1].set_title('Distribution de pledged (log10)')
axes[1].set_xlabel('log10(pledged)')
axes[1].set_ylabel('Fréquence')

# Backers
axes[2].hist(df_clean['backers'], bins=50, edgecolor='black')
axes[2].set_title('Distribution de backers')
axes[2].set_xlabel('Nombre de backers')
axes[2].set_ylabel('Fréquence')

# Age
axes[3].hist(df_clean['age'], bins=30, edgecolor='black')
axes[3].set_title('Distribution de age')
axes[3].set_xlabel('Âge')
axes[3].set_ylabel('Fréquence')


axes[4].set_ylabel('Fréquence')


plt.tight_layout()
plt.savefig('fig2_numeric_distributions.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée: fig2_numeric_distributions.png")
plt.close()

# Variables catégorielles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Category
top_categories = df_clean['category'].value_counts().head(10)
top_categories.plot(kind='barh', ax=axes[0, 0])
axes[0, 0].set_title('Top 10 catégories')
axes[0, 0].set_xlabel('Nombre de projets')

# Country
top_countries = df_clean['country'].value_counts().head(10)
top_countries.plot(kind='barh', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 pays')
axes[0, 1].set_xlabel('Nombre de projets')

# Currency
top_currencies = df_clean['currency'].value_counts().head(10)
top_currencies.plot(kind='barh', ax=axes[1, 0])
axes[1, 0].set_title('Top 10 devises')
axes[1, 0].set_xlabel('Nombre de projets')

# Sex
df_clean['sex'].value_counts().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Distribution par sexe')
axes[1, 1].set_xlabel('Sexe')
axes[1, 1].set_ylabel('Nombre de projets')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('fig3_categorical_distributions.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée: fig3_categorical_distributions.png")
plt.close()

# Relations avec la variable cible
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Success rate by category
success_by_cat = df_clean.groupby('category')['state'].apply(
    lambda x: (x == 'successful').mean()
).sort_values(ascending=False).head(10)
success_by_cat.plot(kind='barh', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Taux de succès par catégorie (Top 10)')
axes[0, 0].set_xlabel('Taux de succès')

# Success rate by country
success_by_country = df_clean.groupby('country')['state'].apply(
    lambda x: (x == 'successful').mean()
).sort_values(ascending=False).head(10)
success_by_country.plot(kind='barh', ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Taux de succès par pays (Top 10)')
axes[0, 1].set_xlabel('Taux de succès')

# Goal by state
df_clean.boxplot(column='goal', by='state', ax=axes[1, 0])
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('Goal selon le state')
axes[1, 0].set_xlabel('État')
axes[1, 0].set_ylabel('Goal (log scale)')
plt.suptitle('')

# Backers by state
df_clean.boxplot(column='backers', by='state', ax=axes[1, 1])
axes[1, 1].set_title('Backers selon le state')
axes[1, 1].set_xlabel('État')
axes[1, 1].set_ylabel('Nombre de backers')
plt.suptitle('')

plt.tight_layout()
plt.savefig('fig4_target_relationships.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée: fig4_target_relationships.png")
plt.close()

# Matrice de corrélation
print("\n Calcul de la matrice de corrélation")

# Encoder state pour la corrélation
df_corr = df_clean.copy()
df_corr['duration_days'] = (df_corr['end_date'] - df_corr['start_date']).dt.days

df_corr['start_year'] = df_corr['start_date'].dt.year

df_corr['start_month'] = df_corr['start_date'].dt.month
df_corr['state_encoded'] = (df_corr['state'] == 'successful').astype(int)
numeric_cols = ['goal', 'pledged', 'backers', 'age', 'duration_days', 
                'start_year', 'start_month', 'state_encoded']
corr_matrix = df_corr[numeric_cols].corr()

plt.figure(figsize=(10, 8))

# Affichage de la matrice
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')

# Bar de couleur
plt.colorbar()

# Labels des axes
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# Valeurs dans les cellules
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig('fig5_correlation_matrix.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée: fig5_correlation_matrix.png")
plt.close()


"""
# Sauvegarder les données nettoyées
df_clean.to_csv('ks-projects-clean.csv', index=False)
print(f"\n Données nettoyées sauvegardées: ks-projects-clean.csv ({df_clean.shape})")"""

print("IDENTIFICATION DES OUTILS DE CLASSIFICATION")

print("SYNTHÈSE DES DONNÉES NETTOYÉES")

print(f"\nShape: {df_clean.shape}")
print(f"\nAttributs finaux:")
for col in df_clean.columns:
    dtype = df_clean[col].dtype
    n_unique = df_clean[col].nunique()
    print(f"  {col:20s}: {str(dtype):10s} - {n_unique:6d} valeurs uniques")

print(f"\nDistribution de la variable cible:")
print(df_clean['state'].value_counts())
print(df_clean['state'].value_counts(normalize=True))

