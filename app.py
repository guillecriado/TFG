import os
import shutil
import pandas as pd
from werkzeug.utils import secure_filename

from flask import Flask, render_template, url_for, request, jsonify, redirect, flash

from src.menu.dataset import Dataset
from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX
from src.neuronalNetworks.neuronal_Network_Pyvis import PyVisNeuronalNetwork
from src.menu import dataset

app = Flask(__name__)
app.secret_key = 'neural_network_app_secret_key'  # For flash messages

# Directory for uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Asegurar que el directorio static/graphs existe
os.makedirs("static/graphs", exist_ok=True)

# Global variables to store our network and dataset
global nx_graph
global pyvis_graph
global current_dataset


# Inicializar la red neuronal si no existe
def initialize_networks():
    global nx_graph, pyvis_graph

    # Comprobar si la red ya ha sido inicializada
    if 'nx_graph' not in globals() or nx_graph is None:
        # Valores predeterminados para la red - ajústalos según tus necesidades
        num_inputs = 3
        num_outputs = 1
        nx_graph = NeuronalNetworkX(num_inputs, num_outputs)
        nx_graph.defaultNetwork()  # Inicializa con la red por defecto

    # Inicializar o actualizar la red PyVis
    if 'pyvis_graph' not in globals() or pyvis_graph is None:
        pyvis_graph = PyVisNeuronalNetwork()

    pyvis_graph.actualise_graph(nx_graph)
    pyvis_graph.generate_HTML()


# Inicializar las redes al inicio
initialize_networks()


@app.route('/')
def index():
    return render_template("dataset_selection.html")


# Route for handling dataset selection from the predefined list
@app.route('/select_dataset', methods=['POST'])
def select_dataset():
    global current_dataset

    data = request.json
    dataset_name = data.get('dataset_name')

    if not dataset_name:
        return jsonify({
            'status': 'error',
            'message': 'No dataset name provided'
        }), 400

    try:
        # Here you would load the specified dataset from scikit-learn
        # For now, we'll just acknowledge the selection

        from sklearn import datasets

        dataset=''

        if dataset_name == 'iris_plants':
            dataset = datasets.load_iris(as_frame=True)
        elif dataset_name == 'diabetes':
            dataset = datasets.load_diabetes(as_frame=True)
        elif dataset_name == 'digits':
            dataset = datasets.load_digits(as_frame=True)
        elif dataset_name == 'linnerrud':
            dataset = datasets.load_linnerud(as_frame=True)
        elif dataset_name == 'wine_recognition':
            dataset = datasets.load_wine(as_frame=True)
        elif dataset_name == 'breast_cancer_diagnostic':
            dataset = datasets.load_breast_cancer(as_frame=True)


        current_dataset = Dataset(dataset.frame, "sklearn")

        return jsonify({
            'status': 'success',
            'message': f'Dataset {dataset_name} selected successfully',
            'redirect': url_for('output_selection')  # Redirect to output selection instead
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Route for handling dataset file uploads
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global current_dataset

    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part in the request'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400

    if file and file.filename.endswith('.csv'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Initialize dataset with the uploaded file
            current_dataset = Dataset(filepath,".csv")

            return jsonify({
                'status': 'success',
                'message': 'File uploaded successfully',
                'redirect': url_for('output_selection')  # Redirect to output selection instead
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    else:
        return jsonify({
            'status': 'error',
            'message': 'Only CSV files are allowed'
        }), 400


@app.route('/output_selection')
def output_selection():
    # New route for output column selection
    return render_template("output_selection.html")


@app.route('/get_dataset_columns')
def get_dataset_columns():
    global current_dataset

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    try:
        columns_data = []

        for column in current_dataset.df.columns:
            # Get column type
            dtype = current_dataset.df[column].dtype
            column_type = str(dtype)

            # Get preview data (first 10 values)
            preview_data = current_dataset.df[column].head(10).tolist()

            columns_data.append({
                'name': column,
                'type': column_type,
                'preview': preview_data
            })

        return jsonify({
            'status': 'success',
            'columns': columns_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/set_target_columns', methods=['POST'])
def set_target_columns():
    global current_dataset

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    data = request.json
    target_columns = data.get('target_columns', [])

    if not target_columns:
        return jsonify({
            'status': 'error',
            'message': 'No target columns selected'
        }), 400

    try:
        # Set the output columns in the dataset
        current_dataset.df_outputs = current_dataset.df[target_columns]

        # Set input dataframe by excluding target columns
        input_columns = [col for col in current_dataset.df.columns if col not in target_columns]
        current_dataset.df_inputs = current_dataset.df[input_columns]

        # Determine the number of outputs for network initialization
        num_outputs = len(target_columns)

        return jsonify({
            'status': 'success',
            'message': 'Target columns set successfully',
            'redirect': url_for('input_selection')  # Redirect to input selection
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/input_selection')
def input_selection():
    # Route for input column selection
    return render_template("input_selection.html")


@app.route('/get_input_columns')
def get_input_columns():
    global current_dataset

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    try:
        columns_data = []

        # Get all columns that are NOT in the target columns
        # First get the names of columns in df_outputs
        output_columns = current_dataset.df_outputs.columns.tolist()

        # Then get only columns that are not in output_columns
        for column in current_dataset.df.columns:
            if column not in output_columns:
                # Get column type
                dtype = current_dataset.df[column].dtype
                column_type = str(dtype)

                # Get preview data (first 10 values)
                preview_data = current_dataset.df[column].head(10).tolist()

                columns_data.append({
                    'name': column,
                    'type': column_type,
                    'preview': preview_data
                })

        return jsonify({
            'status': 'success',
            'columns': columns_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/set_input_columns', methods=['POST'])
def set_input_columns():
    global current_dataset, nx_graph

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    data = request.json
    input_columns = data.get('input_columns', [])

    if not input_columns:
        return jsonify({
            'status': 'error',
            'message': 'No input columns selected'
        }), 400

    try:
        # Update the input columns in the dataset
        current_dataset.df_inputs = current_dataset.df[input_columns]

        # Get the number of inputs and outputs for network initialization
        num_inputs = len(input_columns)
        num_outputs = len(current_dataset.df_outputs.columns)

        # Reinitialize the network with the correct number of inputs and outputs
        nx_graph = NeuronalNetworkX(num_inputs, num_outputs)
        nx_graph.defaultNetwork()

        # Update the PyVis graph
        if 'pyvis_graph' in globals() and pyvis_graph is not None:
            pyvis_graph.actualise_graph(nx_graph)
            pyvis_graph.generate_HTML()

        return jsonify({
            'status': 'success',
            'message': 'Input columns set successfully',
            'redirect': url_for('data_preprocessing')  # Redirect to data preprocessing instead
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/network_design')
def network_design():
    # This would be your next page after input selection
    return render_template("hola.html", cache_buster=os.urandom(8).hex())


# Función para manejar la solicitud de agregar neuronas
@app.route('/add_neuron', methods=['POST'])
def add_neuron():
    global nx_graph, pyvis_graph

    data = request.json
    neuron_count = int(data.get('neuronCount', 0))

    if neuron_count <= 0:
        return jsonify({
            'status': 'error',
            'message': 'La cantidad de neuronas debe ser mayor que 0'
        }), 400

    try:
        # Añadir un nuevo nodo con la cantidad de neuronas especificada
        nx_graph.add_node(neuron_count)

        # Opcional: conectar el nuevo nodo con algunos existentes
        # Esto dependerá de tu lógica específica de conexión
        # Por ejemplo, conectar con la última capa o con nodos específicos

        # Actualizar y generar el gráfico PyVis
        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML()

        return jsonify({
            'status': 'success',
            'message': f'Se han añadido un nodo con {neuron_count} neuronas',
            'neuronCount': neuron_count
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/data_preprocessing')
def data_preprocessing():
    """Route for data preprocessing page"""
    return render_template("data_preprocessing.html")


@app.route('/get_problem_type', methods=['GET'])
def get_problem_type():
    global current_dataset

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    try:
        problem_type = current_dataset.problem_type()
        if problem_type == "ERROR":
            raise Exception ("Problem type error")
        return jsonify({
            'status': 'success',
            'problem_type': problem_type
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/process_data_preprocessing', methods=['POST'])
def process_data_preprocessing():
    global current_dataset

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    data = request.json

    try:
        # Update train_size from train_test_split
        if 'train_test_split' in data:
            current_dataset.train_size = float(data['train_test_split']) / 100.0

        # Process cleaning options here
        cleaning_option = data.get('cleaning_option', '')
        # Add your cleaning logic based on the selected option

        # Process standardization model
        standardization_model = data.get('standardization_model', 'none')
        # Add your standardization logic based on the selected model

        # Process network configuration
        network_type = data.get('network_type', 'empty')

        if network_type == 'standard':
            # Generate standard network with hidden layers and neurons per layer
            hidden_layers = int(data.get('hidden_layers', 1))
            neurons_per_layer = int(data.get('neurons_per_layer', 5))

            # Initialize your network with these parameters
            # e.g., nx_graph = NeuronalNetworkX(num_inputs, num_outputs)
            # nx_graph.create_standard_network(hidden_layers, neurons_per_layer)
        else:
            nx_graph=NeuronalNetworkX(0,0)
            nx_graph.defaultNetwork()
            # Initialize an empty network
            # e.g., nx_graph = NeuronalNetworkX(num_inputs, num_outputs)

        # Redirect to the network design page
        return jsonify({
            'status': 'success',
            'message': 'Data preprocessing settings applied successfully',
            'redirect': url_for('network_design')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)