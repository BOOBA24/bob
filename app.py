import streamlit as st
import pandas as pd
from Methods import Methods

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Data Science App",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personnalisé
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        h1 {
            color: #1E88E5;
            text-align: center;
            padding-bottom: 2rem;
        }
        h2 {
            color: #0D47A1;
            margin-top: 2rem;
        }
        .stSelectbox label {
            color: #333;
            font-weight: bold;
        }
        .stDataFrame {
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    st.title(" Application de Data Science avec Streamlit")

    # Sidebar avec style
    with st.sidebar:
        st.markdown("## Navigation")
        menu = [" Téléchargement de Fichier", " Analyse Exploratoire (EDA)", " Apprentissage Automatique", " Prédiction"]
        choice = st.selectbox("Choisissez une section", menu)

    if choice == " Téléchargement de Fichier":
        st.header(" Téléchargement et Aperçu des Données")
        with st.container():
            st.markdown("""
                ### Instructions
                1. Sélectionnez un fichier CSV à analyser
                2. Visualisez l'aperçu des données
                """)
            df = Methods.telechargement_fichier()
            if df is not None:
                st.session_state['dataframe'] = df
                st.success(" Fichier chargé avec succès!")

    elif choice == " Analyse Exploratoire (EDA)":
        if 'dataframe' in st.session_state:
            st.header(" Analyse Exploratoire des Données")
            tabs = st.tabs([" Statistiques", " Visualisations"])
            
            with tabs[0]:
                Methods.eda(st.session_state['dataframe'])
            
            with tabs[1]:
                Methods.visualisations(st.session_state['dataframe'])
        else:
            st.error(" Veuillez d'abord télécharger un fichier CSV.")

    elif choice == " Apprentissage Automatique":
        if 'dataframe' in st.session_state:
            st.header(" Configuration et Entraînement du Modèle")
            
            col1, col2 = st.columns(2)
            with col1:
                task_type = st.selectbox("Type de tâche", ["classification", "régression"])
            with col2:
                target_column = st.selectbox("Variable cible", st.session_state['dataframe'].columns)
            
            st.session_state['target_column'] = target_column
            
            with st.spinner("Préparation des données en cours..."):
                X_train, X_test, y_train, y_test = Methods.preprocess_data(st.session_state['dataframe'], target_column)
                models, scaler = Methods.train_models(X_train, y_train, task_type)
                
                st.success("✨ Modèles entraînés avec succès!")
                results_df = Methods.evaluate_model(models, scaler, X_test, y_test, task_type)
                
                st.session_state['models'] = models
                st.session_state['scaler'] = scaler
                st.session_state['task_type'] = task_type
                st.session_state['feature_names'] = X_train.columns.tolist()
        else:
            st.error(" Veuillez d'abord télécharger un fichier CSV.")

    elif choice == " Prédiction":
        if 'models' in st.session_state:
            st.header("🎯 Interface de Prédiction")
            
            # Affichage des informations sur le modèle actuel
            st.info(f"""
                📊 Type de tâche : {st.session_state['task_type'].capitalize()}
                🎓 Variable cible : {st.session_state['target_column']}
            """)
            
            # Sélection du modèle avec métriques
            st.subheader("🤖 Sélection du Modèle")
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_model = st.selectbox(
                    "Choisissez un modèle",
                    options=list(st.session_state['models'].keys())
                )
            
            # Interface de saisie des données
            st.subheader("📝 Saisie des Données")
            with st.form("prediction_form"):
                input_data = {}
                cols = st.columns(2)
                
                for i, feature in enumerate(st.session_state['feature_names']):
                    with cols[i % 2]:
                        try:
                            # Pour les caractéristiques numériques
                            series = st.session_state['dataframe'][feature]
                            min_val = float(series.min())
                            max_val = float(series.max())
                            default_value = float(series.mean())
                            
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_value,
                                help=f"Min: {min_val:.2f}, Max: {max_val:.2f}, Moyenne: {default_value:.2f}"
                            )
                        except:
                            # Pour les caractéristiques catégorielles
                            unique_values = st.session_state['dataframe'][feature].unique()
                            if len(unique_values) <= 10:  # Si peu de valeurs uniques, utiliser un selectbox
                                input_data[feature] = st.selectbox(
                                    f"{feature}",
                                    options=unique_values,
                                    help=f"Valeurs possibles: {', '.join(map(str, unique_values))}"
                                )
                            else:  # Sinon, utiliser un text_input
                                input_data[feature] = st.text_input(
                                    f"{feature}",
                                    help=f"Exemple de valeur: {unique_values[0]}"
                                )
                
                submitted = st.form_submit_button("🎯 Prédire", use_container_width=True)
            
            if submitted:
                input_df = pd.DataFrame([input_data])
                result = Methods.prediction(
                    st.session_state['models'],
                    st.session_state['scaler'],
                    input_df,
                    selected_model,
                    st.session_state['task_type']
                )
                
                if result:
                    st.subheader("🎉 Résultat de la Prédiction")
                    if st.session_state['task_type'] == 'classification':
                        st.success(f"Classe prédite : {result['prediction']}")
                        
                        # Affichage des probabilités si disponibles
                        if result['probabilites'] is not None:
                            proba_df = pd.DataFrame({
                                'Classe': range(len(result['probabilites'])),
                                'Probabilité': result['probabilites']
                            })
                            st.write("📈 Probabilités par classe:")
                            st.dataframe(proba_df.style.format({'Probabilité': '{:.2%}'}),
                                        use_container_width=True)
                    else:
                        st.success(f"Valeur prédite : {result['prediction']:.4f}")
        else:
            st.error("⚠️ Veuillez d'abord entraîner un modèle dans l'onglet 'Apprentissage Automatique'.")

if __name__ == '__main__':
    main()
