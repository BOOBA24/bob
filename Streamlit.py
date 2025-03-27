import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Charger les données
fch = 'BeansDataSet.csv'
col = ['Channel', 'Region', 'Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']
Data = pd.read_csv(fch, names=col)

# Convertir les colonnes numériques
numeric_columns = ['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']
Data[numeric_columns] = Data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Configurer Streamlit
st.set_page_config(page_title="Analyse des Données de BeanPods", layout="wide")

# Barre latérale pour la navigation
st.sidebar.title("Menu de Navigation")
menu = st.sidebar.radio("Navigation", ["Accueil", "Aperçu des Données", "Visualisations", "Statistiques"])

# Accueil
if menu == "Accueil":
    st.markdown(
        """
        <div style='text-align:center;'>
            <h1 style='color:#6c63ff;'>Bienvenue sur l'Analyse des Données Beans</h1>
            <p style='color:#4a4a4a;'>Explorez les données et découvrez des insights intéressants à travers des visualisations.</p>
        </div>
        """, unsafe_allow_html=True
    )

# Aperçu des Données
elif menu == "Aperçu des Données":
    st.title("Aperçu des Données")
    
    # Afficher le DataFrame
    st.subheader("Données Brutes")
    st.write(Data)
    
    # Dimensions des données
    st.subheader("Dimensions des Données")
    st.write(f"Nombre de lignes : {Data.shape[0]}")
    st.write(f"Nombre de colonnes : {Data.shape[1]}")
    
    # Valeurs manquantes
    st.subheader("Valeurs Manquantes")
    missing_values = Data.isnull().sum()
    st.write(missing_values)
    st.write(f"Nombre total de valeurs manquantes : {Data.isnull().sum().sum()}")
    
    # Statistiques descriptives
    st.subheader("Statistiques Descriptives")
    st.write(Data.describe())

    # Comptage des catégories
    st.subheader("Comptage des Catégories")
    st.write("**Par Channel**")
    st.write(Data['Channel'].value_counts())
    st.write("**Par Region**")
    st.write(Data['Region'].value_counts())

# Visualisations
elif menu == "Visualisations":
    st.title("Visualisations des Données")

    # Histogrammes
    st.markdown("### Histogrammes des Variables")
    fig, ax = plt.subplots(figsize=(10, 6))
    Data.hist(bins=15, ax=ax)
    st.pyplot(fig)

    # Matrice de corrélation
    st.markdown("### Matrice de Corrélation")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = Data.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.markdown("### Pairplot")
    fig = sns.pairplot(Data, hue='Channel')
    st.pyplot(fig)

# Statistiques
elif menu == "Statistiques":
    st.title("Statistiques et Analyses")
    
    # Boîte à moustaches
    st.markdown("### Boîte à Moustaches")
    fig, ax = plt.subplots(figsize=(10, 6))
    Data.plot(kind='box', ax=ax)
    st.pyplot(fig)

    # Densité des variables
    st.markdown("### Densité des Variables")
    fig, ax = plt.subplots(figsize=(10, 6))
    Data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 6), ax=ax)
    st.pyplot(fig)

    # Matrice de dispersion
    st.markdown("### Matrice de Dispersion")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_matrix(Data, figsize=(10, 6), diagonal='kde', ax=ax)
    st.pyplot(fig)
