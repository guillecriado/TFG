import pandas as pd
from sklearn.model_selection import train_test_split

class dataset:

    def __init__(self, file_name):
        self.path = file_name # TODO: Poner link completo cuando esté el cargador del dataset
        self.df = pd.read_csv(self.path)
        self.df_inputs = self.df
        self.train_size=0.8
        self.random_state=42

    def __show_columns(self):
        print("Columnas disponibles en el dataset:")
        for i, col in enumerate(self.df.columns):
            print(f"{i}: {col}")

    def __columnsselection(self):
        seleccion = input("Introduce los números de las columnas que quieres usar, separados por comas: ")
        indices_seleccionados = [i.strip() for i in seleccion.split(",")]
        print(indices_seleccionados)
        num_inputs = len(indices_seleccionados)
        print(str(num_inputs))
        self.df_inputs = self.df[indices_seleccionados]
        return num_inputs

    def __divide_dataset(self):
        """
        Este método nos servirá para poder dividir el dataset en los conjuntos de entrenamiento y test.
        :param train_size:El porcentaje del dataset que va para entrenamiento
        :param random_state:
        :return:
        """
        df_train, df_test = train_test_split(self.df, train_size=self.train_size, random_state=self.random_state)
        return df_train, df_test

    def divide_input_output(self):
        df_train, df_test = self.__divide_dataset()
        df_train_input=df_train[df_train[self.df_inputs]]
        df_test_input=df_test[df_test[self.df_inputs]]
        # TODO: Averiguar cómo sacar el target.
        return df_train_input, df_test_input

