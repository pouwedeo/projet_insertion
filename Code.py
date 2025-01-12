from src.data_modelisation import DataModel
from  src.data_exploration import  DataExploration
from  src.data_test import  RegressionDiagnostics
import statsmodels.api as sm


#Data exploration
data_exploration = DataExploration()
exploration_show = data_exploration.data_distribution()
years_show = data_exploration.insertion_years()
data_correlation = data_exploration.data_correlation()

#Data modeling
model = DataModel()
model_show = model.model_show()

#Statistique test
X, y = model.data_modeling()
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())
diagnostics = RegressionDiagnostics(ols_model, X, y)
diagnostics.run_all_tests()

