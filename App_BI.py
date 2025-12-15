import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy.stats import chi2_contingency

from decoupage import make_splits
from normalisation import normaliser

#graph de distribution pour chaque vairable et etudier chaque pair de variable avec graphique de correlation



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


print("\n Attribut: currency")
print(df['currency'].value_counts())


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
print(f"Age < 18: {(df['age'] < 18).sum()}")
print(f"Age > 100: {(df['age'] > 100).sum()}")


print(f"\nvaleur de goal < 0: {(df['goal'] < 0).sum()}")
print(f"\nvaleur de pledged < 0: {(df['pledged'] < 0).sum()}")
print(f"\nvaleur de backers < 0: {(df['backers'] < 0).sum()}")
print("\n Attribut: currency ")
print(df['currency'].value_counts().head(10))

print("\n Attributs: goal et pledged ")
print(f"Goal - Min: {df['goal'].min()}, Max: {df['goal'].max()}, Médiane: {df['goal'].median()}")
print(f"Pledged - Min: {df['pledged'].min()}, Max: {df['pledged'].max()}, Médiane: {df['pledged'].median()}")
print(f"Goal = 0: {(df['goal'] == 0).sum()}")
print(f"Pledged = 0: {(df['pledged'] == 0).sum()}")


print("\n Attribut: backers ")
print(df['backers'].describe())
print(f"Backers = 0: {(df['backers'] == 0).sum()}")
print(f"Valeurs manquantes: {df['backers'].isnull().sum()}")

print("\n Attributs: start_date et end_date ")
print(f"Type de start_date: {df['start_date'].dtype}")
print(f"Type de end_date: {df['end_date'].dtype}")
print(f"Dates manquantes: {(df['start_date'].isnull() | df['end_date'].isnull()).sum()}")


print("DÉTECTION DES PROBLÈMES")


colonnes_avec_manquantes = valeurs_manquantes[valeurs_manquantes > 0]

if len(colonnes_avec_manquantes) > 0:
    print(f"Il y a {len(colonnes_avec_manquantes)} attributs avec des valeurs manquantes :")
    for col, nb in colonnes_avec_manquantes.items():
        print(f" - {col} : {nb} valeurs manquantes")

    
if (df['goal'] == 0).sum() > 0:
    print(f"{(df['goal'] == 0).sum()} projets avec goal = 0 (incohérent)")

if df['start_date'].dtype == 'object':
    print("Dates au format texte, on doit convertir en datetime")

if(df['goal'] < 0).sum() > 0:
    print(f"{(df['goal'] < 0).sum()} projets avec goal < 0 (incohérent)")

if(df['pledged'] < 0).sum() > 0:
    print(f"{(df['pledged'] < 0).sum()} projets avec pledged < 0 (incohérent)")

if(df['backers'] < 0).sum() > 0:
    print(f"{(df['backers'] < 0).sum()} projets avec backers < 0 (incohérent)")

if ((df['pledged'] == 0) & (df['backers'] > 0)).sum() > 0:
    print(f"{(df['pledged']==0 and df['backers']>0).sum()} projets avec backers > 0 et pledged = 0 (incohérent)")

if((df['pledged'] !=0) & (df['backers'] == 0)).sum() > 0:
    print(f"{((df['pledged'] != 0) & (df['backers'] == 0)).sum()} projet avec backers = 0 et pledged != 0 (incohérent)")

print("\nValeurs abérrantes des dates")


if(df['start_date'] < '2009-04-28').sum() > 0:
    print(f"{(df['start_date'] < '2009-04-28').sum()} projets avec start_date < 2009-04-28 (incohérent)")

print("\n----------------------")
print("STATISTIQUES DESCRIPTIVES")
state_map = {
    "successful": 1,
    "failed": 0,
    "canceled": 0,
    "suspended": 0,
    "live": None 
}

df['state_num'] = df['state'].map(state_map)

print(df.groupby('subcategory')['state_num'].mean())
print(df.groupby('category')['state_num'].mean())

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
print(f"  Lignes avant filtrage: {df_clean.shape[0]}")
etats_pertinents = ['successful', 'failed','canceled']
df_clean = df_clean[df_clean['state'].isin(etats_pertinents)]
print(f"  Distribution: {df_clean['state'].value_counts()}")

# Filtrer les projets commencant avant 2009-04-28
print('\n on filtre les date qui commencent avant 2009-04-28')
df_clean = df_clean[df_clean['start_date'] > '2009-04-28']

#On filtre les states qui sont undifined
df_clean = df_clean[df_clean['state']!= 'undifined']


#On traite les valeurs manquantes de 'sex' et 'age'
print("\n Traitement des valeurs manquantes")

df_clean['sex'].fillna(df_clean['sex'].mode()[0], inplace=True)


#On convertie des dates
print("\n Conversion des dates")
df_clean['start_date'] = pd.to_datetime(df_clean['start_date'])
df_clean['end_date'] = pd.to_datetime(df_clean['end_date'])

#Suppression des lignes avec des valeurs nulles dans ces colonnes
cols_with_na = df_clean.isnull().sum()[df_clean.isnull().sum() > 0].index

print("Colonnes contenant des valeurs nulles:", list(cols_with_na))
df_clean = df_clean.dropna(subset=cols_with_na)

#Suppression des lignes avec backers = 0 mais pledged > 0
df_clean = df_clean[~((df_clean['backers'] == 0) & (df_clean['pledged'] > 0))]


#conversion des devise en usd
print("\n Conversion des devises")
exchange_rates = {
    'USD': 1.0,
    'GBP': 1.25,  
    'CAD': 0.75,   
    'EUR': 1.08,   
    'AUD': 0.67,  
    'SEK': 0.09,   
    'NZD': 0.60,   
    'DKK': 0.14,   
    'NOK': 0.09,  
    'CHF': 1.10   
}

# Conversion des colonnes goal et pledged
df_clean['goal'] = df_clean['goal'] * df_clean['currency'].map(exchange_rates)
df['pledged'] = df_clean['pledged'] * df_clean['currency'].map(exchange_rates)

print(f"\nTaille finale après nettoyage: {df_clean.shape}")
print(f"  Lignes après filtrage: {df_clean.shape[0]}")

print(df_clean['backers'].describe())

print(df_clean.groupby('state')['backers'].sum())
plt.figure(figsize=(14, 6))

success_rate_country = df_clean.groupby('country')['state'].apply(
    lambda x: (x == 'successful').mean()
).sort_values(ascending=False)

success_rate_country.plot(kind='bar', color='lightgreen')

plt.title("Taux de succès par pays")
plt.xlabel("Pays")
plt.ylabel("Taux de succès")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


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
fig, axes = plt.subplots(2,2, figsize=(15, 10))
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
axes[2].hist(df_clean['backers'], bins=50, range=(0, 500), edgecolor='black')
axes[2].set_title('Distribution de backers (0-500)')
axes[2].set_xlabel('Nombre de backers')
axes[2].set_ylabel('Fréquence')

# Age
axes[3].hist(df_clean['age'], bins=30, edgecolor='black')
axes[3].set_title('Distribution de age')
axes[3].set_xlabel('Âge')
axes[3].set_ylabel('Fréquence')


plt.tight_layout()
plt.savefig('fig2_numeric_distributions.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée: fig2_numeric_distributions.png")
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df_clean.boxplot(column='goal', by='state', ax=axes[0])
axes[0].set_yscale('log')
axes[0].set_title('Goal par état du projet')
axes[0].set_xlabel('State')
axes[0].set_ylabel('Goal (log scale)')

df_clean.boxplot(column='pledged', by='state', ax=axes[1])
axes[1].set_yscale('log')
axes[1].set_title('Pledged par état du projet')
axes[1].set_xlabel('State')
axes[1].set_ylabel('Pledged (log scale)')

plt.suptitle('') 
plt.tight_layout()
plt.savefig('fig_goal_pledged_by_state.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée : fig_goal_pledged_by_state.png")
plt.close()
categorical_cols = ['category', 'subcategory', 'country', 'currency', 'sex']

for col in categorical_cols:
    counts = df_clean[col].value_counts().head(20)  
    plt.figure(figsize=(10, 4))
    counts.plot(kind='bar')
    plt.title(f"Distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'fig_dist_{col}.png', dpi=100, bbox_inches='tight')
    print(f"Figure sauvegardée : fig_dist_{col}.png")
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
axes[1, 1].set_ylim(0, 500) 
plt.suptitle('')

plt.tight_layout()
plt.savefig('fig4_target_relationships.png', dpi=100, bbox_inches='tight')
print("Figure sauvegardée: fig4_target_relationships.png")
plt.close()
numeric_cols = ['age', 'goal', 'pledged', 'backers']

# Calcul de la matrice de corrélation
corr_matrix_numeric = df_clean[numeric_cols].corr(method='kendall')

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix_numeric, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

# Labels sur les axes
plt.xticks(range(len(corr_matrix_numeric.columns)), corr_matrix_numeric.columns, rotation=45)
plt.yticks(range(len(corr_matrix_numeric.columns)), corr_matrix_numeric.columns)

# Valeurs dans les cellules
for i in range(len(corr_matrix_numeric.columns)):
    for j in range(len(corr_matrix_numeric.columns)):
        value = corr_matrix_numeric.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

plt.title("Matrice de corrélation (variables numériques)", fontsize=14)
plt.tight_layout()
plt.savefig("fig_correlation_numeric.png", dpi=110, bbox_inches='tight')
print("Figure sauvegardée : fig_correlation_numeric.png")
plt.close()
plt.figure(figsize=(8,6))

df_clean['duration_days'] = (df_clean['end_date'] - df_clean['start_date']).dt.days
df_clean['duration_days'] = df_clean['duration_days'].clip(lower=1)  # éviter zéro ou négatif

df_clean.boxplot(column='duration_days', by='state')
plt.title("Durée des projets par état")
plt.suptitle('')
plt.xlabel('State')
plt.ylabel('Durée (jours)')
plt.yscale('log')  # optionnel si tu veux compresser les valeurs très longues
plt.tight_layout()
plt.savefig('fig_duration_by_state.png', dpi=100, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))

# Scatter
plt.scatter(df_clean['age'], df_clean['goal'], alpha=0.3, s=15)

# Ligne de tendance linéaire
z = np.polyfit(df_clean['age'], np.log10(df_clean['goal'] + 1), 1)
p = np.poly1d(z)
plt.plot(df_clean['age'], 10**p(df_clean['age']), color='red')

plt.yscale('log')
plt.xlabel("Âge du porteur du projet")
plt.ylabel("Goal (échelle log)")
plt.title("Relation entre l'âge et le goal des projets")
plt.tight_layout()
plt.savefig("fig_age_vs_goal.png", dpi=120)
plt.close()

print("Figure sauvegardée : fig_age_vs_goal.png")
print("Figure sauvegardée : fig_age_vs_goal.png")
print("Figure sauvegardée : fig_duration_by_state.png")

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

# ============================================================
# 5. PRETRAITEMENT
# normalisation
# ============================================================


df_clean_scaled = normaliser(df_clean)


# ============================================================
# 6. DÉCOUPAGE
# Séparation apprentissage / validation / test
# ============================================================


X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
    df_clean_scaled,
    target_col='state_num',   # ou 'success' selon ton nom
    seed=42,
    test_size=0.2,
    val_size=0.2
)


