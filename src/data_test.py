import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from  src.writer import   metric_writer
class RegressionDiagnostics:

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.residuals = y - model.predict(X)

    # Test de normalité des résidus (Shapiro-Wilk et Jarque-Bera)
    def test_normality(self):
        #print("\n--- Test de normalité des résidus ---")
        # Test de Shapiro-Wilk
        shapiro_stat, shapiro_pval = shapiro(self.residuals)
        test_data = {"Shapiro-Wilk Test: ": shapiro_stat," p-value: ":shapiro_pval }
        test_title = '\n\n--- Test de normalité des résidus ---Shapiro-Wilk Test------'
        return  test_title, test_data
        #print(f"Shapiro-Wilk Test: Statistique={shapiro_stat:.3f}, p-value={shapiro_pval:.3e}")


    # Test d'homoscédasticité (Breusch-Pagan)
    def test_homoscedasticity(self):
        print("\n\n\n--- Test d'homoscédasticité ---")

        # Test de Breusch-Pagan
        bp_stat, bp_pval, _, _ = het_breuschpagan(self.residuals, self.X)

        test_data = {"Breusch-Pagan Test: ": bp_stat, " p-value: ": bp_pval}
        test_title = "\n\n----- Test d'homoscédasticité ----"
        return test_title, test_data
        #print(f"Breusch-Pagan Test: Statistique={bp_stat:.3f}, p-value={bp_pval:.3e}")

    # Test d'autocorrélation (Durbin-Watson)
    def test_autocorrelation(self):
        #print("\n--- Test d'autocorrélation ---")

        # Test de Durbin-Watson
        dw_stat = durbin_watson(self.residuals)
        test_data = {"Durbin-Watson Test:: ": dw_stat}
        test_title = "\n\n----- Test d'autocorrélation  ----"
        return test_title, test_data
        #print(f"Durbin-Watson Test: Statistique={dw_stat:.3f}")

    # Test de multicolinéarité (Variance Inflation Factor)
    def test_multicollinearity(self):
        #print("\n--- Test de multicolinéarité (VIF) ---")

        vif_data = pd.DataFrame()
        vif_data["Variable"] = self.X.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, i) for i in range(self.X.shape[1])]
        test_data = {"Durbin-Watson Test:: ": vif_data}
        test_title = "\n\n----- Test de multicolinéarité (VIF)  ----"
        return test_title, test_data
        #print(vif_data)

    # Exécution de tous les tests
    def run_all_tests(self):
        tests = [ self.test_normality(),
                 self.test_homoscedasticity(),
                 self.test_autocorrelation(),
                 self.test_multicollinearity()
                 ]
        title = "\n ============ TESTS STATISTIQUES=============="
        metric_writer('src/metric/tests.csv', tests, title)






# Fonction principale pour effectuer les tests à partir de votre modèle
#def perform_diagnostics(X, y, model):
    #diagnostics = RegressionDiagnostics(model, X, y)
    #diagnostics.run_all_tests()
