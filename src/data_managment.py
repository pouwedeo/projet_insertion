import pandas as pd

class DataManagement:
    #Data import
    data = pd.read_csv("data/database.csv", delimiter=';')

    def __init__(self):
        pass
    #Data view
    def data_show(self):
        print(self.data.head(3))
        print(len(self.data))
        print(self.data.columns())
        print(self.data.info())
        print(self.data.isna().sum())




    #Target column cleaning
    def target_manage(self):

        # Data manage
        # Important features



        selected_columns = [
            'Taux d’insertion', "Taux d'emploi", 'Part des emplois stables',
            'Part des emplois à temps plein', 'Part des emplois de niveau cadre ou profession intermédiaire',
            'Genre', 'Part des femmes', 'Part des diplômés boursiers dans la discipline',
            'Diplôme', 'Disciplines', 'Taux de chômage national', "Année"
        ]

        # Important features selection

        data_select = self.data[selected_columns]

        target_clean = data_select[~data_select['Taux d’insertion'].isin(['ns'])]
        data = target_clean.dropna(subset=['Taux d’insertion'])
        return   data



    #DataFrame cleaning
    def data_manage(self):

       data = self.target_manage()

       for column in data.columns:

           if column in data and not data[column].empty:
               try:
                   if data[column].dtype == 'object':
                       mode_values = data[column].mode()
                       if not mode_values.empty:
                           mode_value = mode_values[0]
                           if mode_value == 'ns':
                               if len(mode_values) > 1:
                                   mode_value = mode_values[1]
                               else:
                                   mode_value = 0
                           data[column] = data[column].replace('ns', mode_value).fillna(mode_value)
                   else:
                       if data[column].isnull().any():
                           mode_value = data[column].mode()[0]
                           data[column] = data[column].fillna(mode_value)
               except Exception as e:
                   print(f"Erreur avec la colonne {column}: {e}")
       return  data