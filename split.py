from sklearn.model_selection import train_test_split
import os
import pandas as pd

# =========================
# SPLIT BRUT (AVANT FEATURES)
# =========================

df = pd.read_csv("data/ks-projects-clean.csv", encoding="ISO-8859-1")

# Conversion dates minimale (ok à faire ici)
df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
df = df.dropna(subset=["start_date", "end_date"])

# Target (peut être modifiée plus tard)
df["target"] = (df["state"] == "successful").astype(int)

# Stratification UNIQUEMENT sur la target
strat = df["target"]

# 60 / 20 / 20
df_temp, df_test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=strat
)

df_train, df_val = train_test_split(
    df_temp,
    test_size=0.25,  # 0.25 × 0.8 = 0.2
    random_state=42,
    stratify=df_temp["target"]
)

# =========================
# SAUVEGARDE DES SPLITS BRUTS
# =========================

os.makedirs("data/raw_splits", exist_ok=True)

df_train.to_csv("data/raw_splits/train.csv", index=False)
df_val.to_csv("data/raw_splits/val.csv", index=False)
df_test.to_csv("data/raw_splits/test.csv", index=False)