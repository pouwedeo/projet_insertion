from  src.data_managment import  DataManagement
import  pandas as pd
import  seaborn as sn
import  matplotlib.pyplot as plt

class DataExploration:
    data_class = DataManagement()
    data = data_class.data_manage()

    #Distribution of 'Taux d’insertion'
    def data_distribution(self):
        data_insertion = pd.to_numeric(self.data['Taux d’insertion'], errors= "coerce")

        plt.figure(figsize=(10, 6))
        sn.histplot(data_insertion, kde= True)
        plt.title('Distribution du Taux d\'Insertion')
        plt.xlabel('Taux d\'Insertion')
        plt.ylabel('Fréquence')
        plt.savefig('src/graphes/distribuution.svg', bbox_inches='tight')
        plt.show()


    #Variable correlation

    def data_correlation(self):
        data_select = self.data[[
            'Taux d’insertion', "Taux d'emploi", 'Part des emplois stables',
            'Part des emplois à temps plein', 'Part des emplois de niveau cadre ou profession intermédiaire',
            'Part des femmes', 'Part des diplômés boursiers dans la discipline',
            'Taux de chômage national', "Année"
        ]]

        data_numeric = data_select.select_dtypes(include=["float64", "int64", "object"])

        data_numeric = data_numeric.apply(pd.to_numeric, errors="coerce")
        data_corr = data_numeric.corr()

        plt.figure(figsize=(10, 6))
        sn.heatmap(data_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matrice de Corrélation")
        plt.savefig('src/graphes/correlation.svg', bbox_inches="tight")
        plt.show()

    #'Taux d’insertion' - years

    def insertion_years(self):

        y = pd.to_numeric(self.data['Taux d’insertion'], errors= "coerce")
        x = self.data['Année']
        plt.figure(figsize=(10, 6))
        sn.lineplot(data=self.data,x=x, y=y)
        plt.title('Répartition du taux d\'insertion par année')
        plt.xlabel('Années')
        plt.ylabel('Taux d\'insertion')
        plt.savefig('src/graphes/Insertion_Annee.svg', bbox_inches='tight')
        plt.show()

