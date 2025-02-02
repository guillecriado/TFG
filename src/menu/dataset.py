import pandas as pd

class dataset:

    def __init__(self, file_name):
        self.path = file_name # TODO: Poner link completo cuando esté el cargador del dataset
        self.df = pd.read_csv(self.path)
        self.df_inputs = self.df

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
