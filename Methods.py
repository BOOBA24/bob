import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Methods:

    @staticmethod
    def telechargement_fichier():
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
        if uploaded_file is not None:
            try:
               
                df = pd.read_csv(uploaded_file)
                
               
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].nunique() / len(df) < 0.5: 
                        df[col] = df[col].astype('category')
                
                
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                st.write("### 📈 Aperçu du fichier")
                st.write(df.head())
                
                
                st.write("### 📑 Types de données")
                dtypes_df = pd.DataFrame({
                    'Colonne': df.dtypes.index,
                    'Type': df.dtypes.values.astype(str)
                })
                st.dataframe(dtypes_df)
                
                return df
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {str(e)}")
                return None
        return None

    @staticmethod
    def eda(df):
        try:
            st.write("### 📈 Statistiques sommaires")
            
            # Aperçu et statistiques de base
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### 📈 Aperçu des données")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.write("#### 📉 Statistiques descriptives")
                # Sélectionner uniquement les colonnes numériques pour describe()
                numeric_df = df.select_dtypes(include=['int64', 'float64'])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                else:
                    st.info("Aucune colonne numérique trouvée dans le jeu de données.")
            
            # Analyse des valeurs manquantes
            st.write("### 🔍 Analyse des valeurs manquantes")
            missing_values = df.isnull().sum()
            missing_pct = (missing_values / len(df) * 100).round(2)
            
            missing_df = pd.DataFrame({
                'Colonne': missing_values.index,
                'Valeurs manquantes': missing_values.values,
                'Pourcentage (%)': missing_pct.values
            })
            
            # Filtrer pour ne montrer que les colonnes avec des valeurs manquantes
            missing_df = missing_df[missing_df['Valeurs manquantes'] > 0]
            
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
                
                # Créer un graphique des valeurs manquantes
                fig = px.bar(
                    missing_df,
                    x='Colonne',
                    y='Pourcentage (%)',
                    title='Pourcentage de valeurs manquantes par colonne',
                    color='Pourcentage (%)',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ Aucune valeur manquante détectée dans le jeu de données!")
            
            # Distribution des types de données
            st.write("### 📓 Types de données")
            dtypes_counts = df.dtypes.value_counts()
            dtypes_df = pd.DataFrame({
                'Type': dtypes_counts.index.astype(str),
                'Nombre de colonnes': dtypes_counts.values
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(dtypes_df, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    dtypes_df,
                    values='Nombre de colonnes',
                    names='Type',
                    title='Distribution des types de données'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Détails des colonnes
            st.write("### 📋 Détails des colonnes")
            details_df = pd.DataFrame({
                'Colonne': df.columns,
                'Type': df.dtypes.astype(str),
                'Valeurs uniques': [df[col].nunique() for col in df.columns],
                'Valeurs manquantes (%)': [(df[col].isnull().sum() / len(df) * 100).round(2) for col in df.columns]
            })
            st.dataframe(details_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse exploratoire : {str(e)}")
            st.write("Détails de l'erreur pour le débogage :")
            st.code(str(e))

    @staticmethod
    def visualisations(df):
        try:
            # Configuration du thème des graphiques
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Séparation des variables numériques et catégorielles
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0:
                st.write("### 📈 Distributions des Variables Numériques")
                
                # Sélection des variables à visualiser
                if len(numeric_cols) > 1:
                    selected_numeric = st.multiselect(
                        "Sélectionnez les variables numériques à visualiser",
                        numeric_cols,
                        default=list(numeric_cols)[:2]
                    )
                else:
                    selected_numeric = numeric_cols
                
                if selected_numeric:
                    for i, column in enumerate(selected_numeric):
                        # Création de la figure avec subplot pour histogramme et box plot
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogramme
                            fig_hist = px.histogram(
                                df,
                                x=column,
                                color_discrete_sequence=[colors[i % len(colors)]],
                                title=f"Distribution de {column}",
                                template="simple_white"
                            )
                            
                            fig_hist.update_layout(
                                showlegend=False,
                                title_x=0.5,
                                title_font_size=16,
                                height=400
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Box plot
                            fig_box = px.box(
                                df,
                                y=column,
                                color_discrete_sequence=[colors[i % len(colors)]],
                                title=f"Box Plot de {column}",
                                template="simple_white"
                            )
                            
                            fig_box.update_layout(
                                showlegend=False,
                                title_x=0.5,
                                title_font_size=16,
                                height=400
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Statistiques descriptives
                        stats = df[column].describe()
                        st.write(f"**Statistiques pour {column}:**")
                        st.write({
                            "Moyenne": f"{stats['mean']:.2f}",
                            "Médiane": f"{stats['50%']:.2f}",
                            "Écart-type": f"{stats['std']:.2f}",
                            "Min": f"{stats['min']:.2f}",
                            "Max": f"{stats['max']:.2f}"
                        })
                        st.markdown("---")
            
            if len(cat_cols) > 0:
                st.write("### 📁 Distribution des Variables Catégorielles")
                
                # Sélection des variables catégorielles
                if len(cat_cols) > 1:
                    selected_cat = st.multiselect(
                        "Sélectionnez les variables catégorielles à visualiser",
                        cat_cols,
                        default=list(cat_cols)[:2]
                    )
                else:
                    selected_cat = cat_cols
                
                if selected_cat:
                    for column in selected_cat:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Diagramme en barres
                            value_counts = df[column].value_counts()
                            fig_bar = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Distribution de {column}",
                                labels={'x': column, 'y': 'Nombre'},
                                template="simple_white"
                            )
                            fig_bar.update_layout(title_x=0.5, height=400)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        with col2:
                            # Camembert
                            fig_pie = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title=f"Proportion de {column}",
                                template="simple_white"
                            )
                            fig_pie.update_layout(title_x=0.5, height=400)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Affichage des statistiques
                        st.write(f"**Fréquences pour {column}:**")
                        freq_df = pd.DataFrame({
                            'Valeur': value_counts.index,
                            'Nombre': value_counts.values,
                            'Pourcentage (%)': (value_counts.values / len(df) * 100).round(2)
                        })
                        st.dataframe(freq_df, use_container_width=True)
                        st.markdown("---")
            
            if len(numeric_cols) >= 2:
                st.write("### 📉 Matrice de Corrélation")
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title="Matrice de Corrélation des Variables Numériques"
                )
                fig_corr.update_layout(title_x=0.5)
                st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors de la création des visualisations : {str(e)}")
            st.write("Détails de l'erreur pour le débogage :")
            st.code(str(e))

    @staticmethod
    def preprocess_data(df, target_column):
        try:
            # Séparation des features et de la cible
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Gestion des valeurs manquantes
            numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            
            # Remplacement des valeurs manquantes
            for col in numeric_columns:
                X[col] = X[col].fillna(X[col].mean())
            for col in categorical_columns:
                X[col] = X[col].fillna(X[col].mode()[0])
            
            # Encodage des variables catégorielles
            X = pd.get_dummies(X, drop_first=True)
            
            # Division en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            st.error(f"Erreur lors du prétraitement des données : {str(e)}")
            return None, None, None, None

    @staticmethod
    def train_models(X_train, y_train, task_type):
        models = {}
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        with st.spinner('Entraînement des modèles en cours...'):
            if task_type == 'classification':
                models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
                models['Logistic Regression'] = LogisticRegression(random_state=42)
                models['SVM'] = SVC(random_state=42)
                models['KNN'] = KNeighborsClassifier(n_neighbors=5)
            else:
                models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
                models['Linear Regression'] = LinearRegression()
                models['SVR'] = SVR()
                models['KNN'] = KNeighborsRegressor(n_neighbors=5)
            
            for name, model in models.items():
                with st.spinner(f'Entraînement du modèle {name}...'):
                    model.fit(X_train_scaled, y_train)
        
        return models, scaler

    @staticmethod
    def evaluate_model(models, scaler, X_test, y_test, task_type):
        X_test_scaled = scaler.transform(X_test)
        results = []

        with st.spinner('⏳ Évaluation des modèles en cours...'):
            for name, model in models.items():
                with st.spinner(f'📊 Évaluation du modèle {name}...'):
                    y_pred = model.predict(X_test_scaled)
                    
                    if task_type == 'classification':
                        metrics = {
                            'Modèle': name,
                            'Précision (Accuracy)': round(accuracy_score(y_test, y_pred), 4),
                            'Rappel': round(recall_score(y_test, y_pred, average='weighted'), 4),
                            'Précision (Precision)': round(precision_score(y_test, y_pred, average='weighted'), 4),
                            'F1-Score': round(f1_score(y_test, y_pred, average='weighted'), 4)
                        }
                    else:
                        metrics = {
                            'Modèle': name,
                            'MAE': round(mean_absolute_error(y_test, y_pred), 4),
                            'MSE': round(mean_squared_error(y_test, y_pred), 4),
                            'R²': round(r2_score(y_test, y_pred), 4)
                        }
                    
                    results.append(metrics)
            
            results_df = pd.DataFrame(results)
            
            # Affichage des métriques avec un style amélioré
            st.write("### 📊 Métriques de Performance")
            if task_type == 'classification':
                st.markdown("""
                    - **Précision (Accuracy)** : Pourcentage de prédictions correctes
                    - **Rappel** : Capacité à identifier tous les cas positifs
                    - **Précision (Precision)** : Exactitude des prédictions positives
                    - **F1-Score** : Moyenne harmonique de la précision et du rappel
                """)
            else:
                st.markdown("""
                    - **MAE** : Erreur moyenne absolue
                    - **MSE** : Erreur quadratique moyenne
                    - **R²** : Coefficient de détermination (0 à 1, 1 étant parfait)
                """)
            
            st.dataframe(
                results_df.style
                    .highlight_max(axis=0, subset=results_df.columns[1:], props='color:green')
                    .format(precision=4),
                use_container_width=True
            )
            
            # Afficher le meilleur modèle
            if task_type == 'classification':
                best_metric = 'F1-Score'
            else:
                best_metric = 'R²'
            best_model = results_df.loc[results_df[best_metric].idxmax(), 'Modèle']
            st.success(f"✨ Meilleur modèle basé sur {best_metric}: {best_model}")
        
        return results_df

    @staticmethod
    def prediction(models, scaler, input_data, selected_model, task_type):
        try:
            # Vérification des données d'entrée
            if not all(str(v).strip() for v in input_data.values):
                st.error('⚠️ Certaines valeurs sont manquantes. Veuillez remplir tous les champs.')
                return None

            # Conversion des données en nombres si possible
            for col in input_data.columns:
                try:
                    input_data[col] = pd.to_numeric(input_data[col])
                except (ValueError, TypeError):
                    pass  # Garde la colonne comme chaîne de caractères si la conversion échoue

            with st.spinner('🔍 Préparation des données...'):
                # Mise à l'échelle des données
                input_scaled = scaler.transform(input_data)
                model = models[selected_model]

            with st.spinner('🤖 Calcul de la prédiction...'):
                prediction = model.predict(input_scaled)
                
                # Formatage de la prédiction selon le type de tâche
                if task_type == 'classification':
                    result = prediction[0]
                    proba = None
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_scaled)[0]
                    return {'prediction': result, 'probabilites': proba}
                else:  # régression
                    return {'prediction': float(prediction[0]), 'probabilites': None}

        except Exception as e:
            st.error(f'⚠️ Erreur lors de la prédiction : {str(e)}')
            return None

    @staticmethod
    def choix_sidebar():
        return Methods  # Retourne la classe Methods elle-même pour appeler les méthodes
