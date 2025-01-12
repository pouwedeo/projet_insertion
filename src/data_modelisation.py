import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from src.writer import metric_writer
from  src.data_managment import DataManagement

class DataModel:

    #instanciation
    data = DataManagement()
    clean_data = data.data_manage()

    #Data preparation
    def data_modeling(self):
        # Séparation des features et de la cible
        X = self.clean_data.drop(columns=['Taux d’insertion'])
        y = self.clean_data['Taux d’insertion']

        # Identification des colonnes catégoriques
        categorical_cols = ['Genre', 'Diplôme', 'Disciplines']
        categorical_cols = [col for col in categorical_cols if col in X.columns]

        # Encodage des colonnes catégoriques
        if categorical_cols:
            encoder = OneHotEncoder(drop="first", sparse_output=False)
            X_encoded = pd.DataFrame(
                encoder.fit_transform(X[categorical_cols]),
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X.index  # Maintenir les indices alignés
            )
            X = X.drop(columns=categorical_cols)
            X = pd.concat([X, X_encoded], axis=1)

        # Imputation des valeurs manquantes
        final_imputer = SimpleImputer(strategy="most_frequent")
        X = pd.DataFrame(final_imputer.fit_transform(X), columns=X.columns, index=X.index)

        # Conversion des données en numérique
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # Remplacement des NaN restants par la moyenne
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Suppression des colonnes constantes
        X = X.loc[:, (X != X.iloc[0]).any()]

        # Ajout d'une constante pour l'interpolation
        X = sm.add_constant(X)

        # Réaligner explicitement les indices
        X, y = X.align(y, join="inner", axis=0)

        return X, y

    # Regression model train


    def model_show(self):
        try:
            # Préparation des données
            X, y = self.data_modeling()

            # Ajustement du modèle
            model = sm.OLS(y, X).fit()

            # Résumé du modèle
            title = "Résumé du modèle"
            model_summary = model.summary()
            metric_writer('src/metric/model_summary.csv',model_summary.as_text(), title)

            # Prédictions
            y_pred = model.predict(X)

            # Graphique 1 : Valeurs Observées vs Prédites
            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred, color='blue', alpha=0.6, label='Prédictions')
            plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ligne parfaite')
            plt.title('Valeurs Observées vs. Prédites', fontsize=14)
            plt.xlabel('Valeurs Observées', fontsize=12)
            plt.ylabel('Valeurs Prédites', fontsize=12)
            plt.legend(title=f'R² : {model.rsquared_adj:.2f}')
            plt.grid(True)
            plt.savefig('src/graphes/regression.svg', bbox_inches="tight")
            plt.show()

            # Graphique 2 : Distribution des Résidus
            residuals = y - y_pred
            plt.figure(figsize=(10, 6))
            sns.kdeplot(residuals, fill=True, color='skyblue', label='Densité des résidus')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
            plt.title('Distribution des Résidus (KDE)', fontsize=14)
            plt.xlabel('Résidus (Valeurs Observées - Prédites)', fontsize=12)
            plt.ylabel('Densité', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.savefig('src/graphes/kde.svg', bbox_inches='tight')
            plt.show()

            # Graphique 3 : Résidus vs. Valeurs Prédites
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
            plt.title('Résidus vs. Valeurs Prédites', fontsize=14)
            plt.xlabel('Valeurs Prédites', fontsize=12)
            plt.ylabel('Résidus', fontsize=12)
            plt.grid(True)
            plt.savefig('src/graphes/Résidus.svg', bbox_inches='tight')
            plt.show()

            # Métriques supplémentaires
            print("\n--- Métriques supplémentaires ---")

            title_r ="R² ajusté"
            metric_r = f"{model.rsquared_adj:.2f}"
            metric_writer('src/metric/rsquared.csv',metric_r, title_r)
            error_rmse = f"{((residuals ** 2).mean()) ** 0.5:.3f}"
            error_title = "Racine carrée de l'erreur quadratique moyenne (RMSE)"
            metric_writer('src/metric/RMSE.csv', error_rmse, error_title)

        except Exception as e:
            print(f"Erreur lors de l'exécution de model_show : {e}")
