import pandas as pd

def imputer_valeurs_manquantes(df):
    return df.fillna(df.mean())

def encoder_variables_categorielles(df):
    return pd.get_dummies(df)
