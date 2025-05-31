import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, file_name, extension):
        if(extension == '.csv'):
            self.path = file_name
            self.df = pd.read_csv(self.path)
        elif(extension == 'sklearn'):
            self.path = ""
            self.df = file_name
        self.df_inputs = self.df
        self.df_outputs = self.df
        self.train_size=0.8
        self.random_state=42
        self.test_size=1-self.train_size
        self.standarization=''

    def input_columns_selection(self):
        seleccion = input("Introduce los números de las columnas que quieres usar como input, separados por comas: ")
        indices_seleccionados = [i.strip() for i in seleccion.split(",")]
        print(indices_seleccionados)
        num_inputs = len(indices_seleccionados)
        print(str(num_inputs))
        self.df_inputs = self.df[indices_seleccionados]
        return num_inputs

    def target_columns_selection(self):
        seleccion = input("Introduce los números de las columnas que quieres usar como target, separados por comas: ")
        indices_seleccionados = [i.strip() for i in seleccion.split(",")]
        print(indices_seleccionados)
        num_outputs = len(indices_seleccionados)
        print(str(num_outputs))
        self.df_outputs = self.df[indices_seleccionados]
        return num_outputs

    def problem_type(self):
        """
        Este método nos sirve para determinar ante qué problema nos enfrentamos a través del tipo de datos que contenga la columna "target".
        En caso de que sea numérico y contenga más de 10 valores únicos consideramos que es un problema de regresión, en caso de que sea menor, lo tomamos como una clasificación binaria.
        En caso de que no sea numérico y sea o de valor categórico (dtype=object) o string, lo consideraremos que es un problema de clasificación.
        En caso de que no sea ninguno de estos dos, se generará un error.
        :return: El tipo de problema al que nos enfrentamos.
        """
        type=""
        if pd.api.types.is_numeric_dtype(self.df_outputs):
            if self.df_outputs.nunique() >= 10:
                type='REGRESSION'
            else:
                type='BINARY CLASSIFICATION'
        elif isinstance(self.df_outputs, pd.CategoricalDtype) or pd.api.types.is_string_dtype(self.df_outputs):
            type='CLASSIFICATION'
        else:
            type="ERROR"

        return type

    def __divide_dataset(self):
        """
        Este método nos servirá para poder dividir el dataset en los conjuntos de entrenamiento y test.
        :return: Los conjuntos de datos del dataset para entrenamiento y test de la red neuronal.
        """
        df_train, df_test = train_test_split(self.df, train_size=self.train_size, random_state=self.random_state)
        return df_train, df_test

    def divide_data(self):
        """
        Este método nos sirve para dividir los conjuntos de input y output de cada conjunto
        :return: Los conjuntos completamente divididos para el entrenamiento y test de la red neuronal.
        """
        df_train, df_test = self.__divide_dataset()
        df_train_input=df_train[df_train[self.df_inputs]]
        df_test_input=df_test[df_test[self.df_inputs]]
        df_train_output=df_train[df_train[self.df_outputs]]
        df_test_output=df_test[df_test[self.df_outputs]]
        return df_train_input, df_test_input, df_train_output, df_test_output

    def set_train_test_size(self, trainSize):
        self.train_size=trainSize
        self.test_size=1-self.train_size

    def set_standarization(self, standarization):
        self.standarization=standarization

    def cleaning(self,cleaning_option,custom_value):
        if cleaning_option == 'row-null':
            self.df.dropna(inplace=True)
        elif cleaning_option == 'column-null':
            self.df.dropna(axis=1, inplace=True)
        elif cleaning_option == 'custom':
            self.df.replace(custom_value, np.nan, inplace=True)
            self.df.dropna(inplace=True)
            # Código alternativo
            # df.drop(df[df.isin([custom_value]).any(axis=1)].index, inplace=True)
