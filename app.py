import os
import shutil
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

        # TODO: Replace with actual scikit-learn dataset loading
        # from sklearn import datasets
        # if dataset_name == 'olivetti_faces':
        #     dataset = datasets.fetch_olivetti_faces()
        # elif dataset_name == '20newsgroups':
        #     dataset = datasets.fetch_20newsgroups()
        # ...

        # Placeholder for dataset object
        # current_dataset = Dataset('path/to/dataset_file.csv')

        return jsonify({
            'status': 'success',
            'message': f'Dataset {dataset_name} selected successfully',
            'redirect': url_for('network_design')
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
            current_dataset = Dataset(filepath)

            return jsonify({
                'status': 'success',
                'message': 'File uploaded successfully',
                'redirect': url_for('network_design')
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


@app.route('/network_design')
def network_design():
    # This would be your next page after dataset selection
    # For now, we'll just redirect to the hola.html page
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


if __name__ == "__main__":
    app.run(debug=True)