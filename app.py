import os
import shutil
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename

from flask import Flask, render_template, url_for, request, jsonify, redirect, flash

from src.menu.dataset import Dataset
from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX
from src.neuronalNetworks.neuronal_Network_Pyvis import PyVisNeuronalNetwork
from src.menu import dataset

app = Flask(__name__)
app.secret_key = 'neural_network_app_secret_key'  # For flash messages

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Deshabilitar caché para archivos estáticos

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

nx_graph = None
pyvis_graph = None
current_dataset = None


# Inicializar la red neuronal si no existe
def initialize_networks():
    global nx_graph, pyvis_graph
    if pyvis_graph is None:
        pyvis_graph = PyVisNeuronalNetwork()
    if nx_graph is not None:
        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML(nx_graph)





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
        if current_dataset.problem_type() == 'CLASSIFICATION':
            num_outputs = int(current_dataset.df_outputs.nunique()[0])
        else: #Si no es clasificación solo tendremos un output ya que no hay que predecir clases
            num_outputs = len(current_dataset.df_outputs.columns)


        nx_graph = NeuronalNetworkX(num_inputs, num_outputs)
        #nx_graph.defaultNetwork(0, 0)

        # Update the PyVis graph
        if 'pyvis_graph' in globals() and pyvis_graph is not None:
            pyvis_graph.actualise_graph(nx_graph)
            pyvis_graph.generate_HTML(nx_graph)

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
    global nx_graph, pyvis_graph

    # Initialize pyvis_graph if it doesn't exist
    if pyvis_graph is None:
        pyvis_graph = PyVisNeuronalNetwork()

    # Generate HTML if nx_graph exists
    if nx_graph is not None:
        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML(nx_graph)

    return render_template("manage_neuronal_network.html", cache_buster=os.urandom(8).hex())

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
        nx_graph.add_node(neuron_count,'Hidden Neuron')

        # Opcional: conectar el nuevo nodo con algunos existentes
        # Esto dependerá de tu lógica específica de conexión
        # Por ejemplo, conectar con la última capa o con nodos específicos

        # Actualizar y generar el gráfico PyVis
        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML(nx_graph)

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
    global current_dataset, nx_graph, pyvis_graph  # Add pyvis_graph to globals

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    data = request.json

    try:
        nx_graph.set_problem_type(current_dataset.problem_type())
        # Update train_size from train_test_split
        if 'train_test_split' in data:
            current_dataset.set_train_test_size((float(data['train_test_split']) / 100.0))

        # Process cleaning options here
        cleaning_option = data.get('cleaning_option', '')
        if cleaning_option == '':
            current_dataset.cleaning(None, '')
        elif cleaning_option == 'custom':
            custom_value = data.get('custom_value', '')
            current_dataset.cleaning(cleaning_option, custom_value)
        else:
            current_dataset.cleaning(cleaning_option, '')
        # Process standardization model
        standardization_model = data.get('standardization_model', 'none')
        if standardization_model != 'none':
            current_dataset.set_standarization(standardization_model)
        # Process network configuration
        network_type = data.get('network_type', 'empty')

        if network_type == 'standard':
            # Generate standard network with hidden layers and neurons per layer
            hidden_layers = int(data.get('hidden_layers', 1))
            neurons_per_layer = int(data.get('neurons_per_layer', 5))
            nx_graph.defaultNetwork(hidden_layers, neurons_per_layer)
        else:
            nx_graph.defaultNetwork(0, 0)

        # Initialize pyvis_graph if it doesn't exist
        if pyvis_graph is None:
            pyvis_graph = PyVisNeuronalNetwork()

        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML(nx_graph)

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


@app.route('/add_edge', methods=['POST'])
def add_edge():
    global nx_graph, pyvis_graph

    data = request.json
    source = int(data.get('source', -1))
    target = int(data.get('target', -1))

    if source < 0 or target < 0:
        return jsonify({
            'status': 'error',
            'message': 'Invalid node IDs. Source and target must be non-negative.'
        }), 400

    try:
        # Check if nodes exist
        if source not in nx_graph.nxg.nodes():
            return jsonify({
                'status': 'error',
                'message': f'Source node {source} does not exist'
            }), 400

        if target not in nx_graph.nxg.nodes():
            return jsonify({
                'status': 'error',
                'message': f'Target node {target} does not exist'
            }), 400

        # Check if edge already exists
        if nx_graph.nxg.has_edge(source, target):
            return jsonify({
                'status': 'error',
                'message': f'Edge from {source} to {target} already exists'
            }), 400

        # Add the edge
        nx_graph.add_edge(source, target)

        # Update and generate the PyVis graph
        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML(nx_graph)

        return jsonify({
            'status': 'success',
            'message': f'Edge added from node {source} to node {target}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/delete_node', methods=['POST'])
def delete_node():
    global nx_graph, pyvis_graph

    data = request.json
    node_id = int(data.get('node_id', -1))

    if node_id < 0:
        return jsonify({
            'status': 'error',
            'message': 'Invalid node ID. Node ID must be non-negative.'
        }), 400

    try:
        # Check if it's an input or output node (these shouldn't be deleted)
        if node_id in nx_graph.inputNeurons:
            return jsonify({
                'status': 'error',
                'message': 'Cannot delete input neurons'
            }), 400

        if node_id in nx_graph.outputNeurons:
            return jsonify({
                'status': 'error',
                'message': 'Cannot delete output neurons'
            }), 400

        # Remove the node (this will also remove all its edges automatically)
        success = nx_graph.remove_node(node_id)

        if not success:
            return jsonify({
                'status': 'error',
                'message': f'Node {node_id} does not exist'
            }), 400

        # Update and generate the PyVis graph
        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML(nx_graph)

        return jsonify({
            'status': 'success',
            'message': f'Node {node_id} and all its edges have been removed'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/delete_edge', methods=['POST'])
def delete_edge():
    global nx_graph, pyvis_graph

    data = request.json
    source = int(data.get('source', -1))
    target = int(data.get('target', -1))

    if source < 0 or target < 0:
        return jsonify({
            'status': 'error',
            'message': 'Invalid node IDs. Source and target must be non-negative.'
        }), 400

    try:
        # Remove the edge
        success = nx_graph.remove_edge(source, target)

        if not success:
            return jsonify({
                'status': 'error',
                'message': f'Edge from {source} to {target} does not exist'
            }), 400

        # Update and generate the PyVis graph
        pyvis_graph.actualise_graph(nx_graph)
        pyvis_graph.generate_HTML(nx_graph)

        return jsonify({
            'status': 'success',
            'message': f'Edge from node {source} to node {target} has been removed'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/check_network_connectivity', methods=['GET'])
def check_network_connectivity():
    """Check if the neural network is fully connected"""
    global nx_graph

    if nx_graph is None:
        return jsonify({
            'status': 'error',
            'message': 'No neural network has been created yet'
        }), 400

    try:
        # Check if network is fully connected
        is_fully_connected_td = nx_graph.isFullyConnectedTopDown()
        is_fully_connected_bu = nx_graph.isFullyConnectedBottomUp()

        is_fully_connected = is_fully_connected_td and is_fully_connected_bu

        if is_fully_connected:
            return jsonify({
                'status': 'success',
                'is_fully_connected': True,
                'message': 'Network is fully connected and ready for training'
            })
        else:
            return jsonify({
                'status': 'error',
                'is_fully_connected': False,
                'message': 'Network is not fully connected. Please ensure all layers are properly connected before training.'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error checking network connectivity: {str(e)}'
        }), 500


@app.route('/train_network', methods=['POST'])
def train_network():
    """Train the neural network"""
    global nx_graph, current_dataset

    if nx_graph is None:
        return jsonify({
            'status': 'error',
            'message': 'No neural network has been created yet'
        }), 400

    if current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset has been loaded yet'
        }), 400

    data = request.json
    num_epochs = int(data.get('epochs', 100))
    optimizer = data.get('optimizer', 'adam')
    learning_rate = float(data.get('learning_rate', 0.001))

    try:
        nx_graph.set_optimizer(optimizer, learning_rate)
        # First check if network is fully connected
        is_fully_connected_td = nx_graph.isFullyConnectedTopDown()
        is_fully_connected_bu = nx_graph.isFullyConnectedBottomUp()

        if not (is_fully_connected_td and is_fully_connected_bu):
            return jsonify({
                'status': 'error',
                'message': 'Network is not fully connected. Cannot train.'
            }), 400

        # Set the loss function based on problem type
        problem_type = current_dataset.problem_type()
        nx_graph.set_problem_type(problem_type)
        nx_graph.set_loss(problem_type)

        # Parse the network to Keras
        nx_graph.parseKeras()

        # Divide the dataset
        df_train_input, df_test_input, df_train_output, df_test_output = current_dataset.divide_data()
        # Train the model
        nx_graph.train_model(df_train_input, df_train_output, num_epochs)

        # Test the model
        test_results = nx_graph.test_model(df_test_input, df_test_output)

        return jsonify({
            'status': 'success',
            'message': f'Model trained successfully for {num_epochs} epochs',
            'test_loss': float(test_results) if isinstance(test_results, (int, float)) else float(test_results[0]),
            'model_saved': True,
            'model_path': nx_graph.keras_path
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during training: {str(e)}'
        }), 500


@app.route('/get_training_info', methods=['GET'])
def get_training_info():
    """Get information about the dataset and network for training"""
    global nx_graph, current_dataset

    if nx_graph is None or current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'Network or dataset not properly initialized'
        }), 400

    try:
        # Get dataset info
        num_samples = len(current_dataset.df)
        num_inputs = len(current_dataset.df_inputs.columns)
        num_outputs = len(current_dataset.df_outputs.columns)
        problem_type = current_dataset.problem_type()
        train_size = int(current_dataset.train_size * 100)

        # Get network info
        num_nodes = nx_graph.nxg.number_of_nodes()
        num_edges = nx_graph.nxg.number_of_edges()

        return jsonify({
            'status': 'success',
            'dataset_info': {
                'num_samples': num_samples,
                'num_inputs': num_inputs,
                'num_outputs': num_outputs,
                'problem_type': problem_type,
                'train_split': train_size,
                'test_split': 100 - train_size
            },
            'network_info': {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_hidden_layers': num_nodes - num_inputs - num_outputs
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500




@app.route('/results')
def results():
    """Route for the results page"""
    return render_template("results.html")


@app.route('/get_results_data')
def get_results_data():
    """Get data needed for the results page"""
    global current_dataset, nx_graph

    if current_dataset is None or nx_graph is None:
        return jsonify({
            'status': 'error',
            'message': 'No dataset or network available'
        }), 400

    try:
        # Get input and output columns
        input_columns = current_dataset.df_inputs.columns.tolist()
        output_columns = current_dataset.df_outputs.columns.tolist()
        problem_type = current_dataset.problem_type()

        # Get test data for evaluation
        df_train_input, df_test_input, df_train_output, df_test_output = current_dataset.divide_data()

        # Calculate metrics
        predictions = nx_graph.predict(df_test_input)


        metrics = {}
        if problem_type == 'REGRESSION':
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np

            mae = mean_absolute_error(df_test_output, predictions)
            mse = mean_squared_error(df_test_output, predictions)
            rmse = np.sqrt(mse)

            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse
            }
        else:
            from sklearn.metrics import accuracy_score, recall_score, f1_score
            import numpy as np

            # Convert predictions to class predictions
            if problem_type == 'BINARY CLASSIFICATION':
                pred_classes = (predictions > 0.5).astype(int)
            else:
                pred_classes = np.argmax(predictions, axis=1)

            accuracy = accuracy_score(df_test_output, pred_classes)
            recall = recall_score(df_test_output, pred_classes, average='weighted')
            f1 = f1_score(df_test_output, pred_classes, average='weighted')

            metrics = {
                'accuracy': accuracy,
                'recall': recall,
                'f1_score': f1
            }

        return jsonify({
            'status': 'success',
            'input_columns': input_columns,
            'output_columns': output_columns,
            'problem_type': problem_type,
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/make_predictions', methods=['POST'])
def make_predictions():
    """Make predictions with the trained model"""
    global nx_graph

    if nx_graph is None:
        return jsonify({
            'status': 'error',
            'message': 'No trained model available'
        }), 400

    try:
        data = request.json
        input_data = data.get('input_data', [])

        if not input_data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400

        # Make predictions
        predictions = nx_graph.predict(input_data)
        best_prediction= str(np.argmax(predictions))
        print(predictions)
        return jsonify({
            'status': 'success',
            'predictions': best_prediction
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/download_model')
def download_model():
    """Download the trained model"""
    global nx_graph

    if nx_graph is None:
        return jsonify({
            'status': 'error',
            'message': 'No trained model available'
        }), 400

    try:
        from flask import send_file
        import os

        # Check if model file exists
        if not os.path.exists(nx_graph.keras_path):
            return jsonify({
                'status': 'error',
                'message': 'Model file not found'
            }), 404

        # Send the model file as download
        return send_file(
            nx_graph.keras_path,
            as_attachment=True,
            download_name='neural_network_model.keras',
            mimetype='application/octet-stream'
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/save_model_action', methods=['POST'])
def save_model_action():
    """Save model action for the results page"""
    global nx_graph

    if nx_graph is None:
        return jsonify({
            'status': 'error',
            'message': 'No trained model available'
        }), 400

    try:
        # The model should already be saved after training, but let's ensure it
        nx_graph.save_model()

        return jsonify({
            'status': 'success',
            'message': 'Model saved successfully',
            'download_url': url_for('download_model')
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/evaluate_model_action', methods=['POST'])
def evaluate_model_action():
    """Evaluate model action for the results page"""
    global nx_graph, current_dataset

    if nx_graph is None or current_dataset is None:
        return jsonify({
            'status': 'error',
            'message': 'No trained model or dataset available'
        }), 400

    try:
        # Get test data
        df_train_input, df_test_input, df_train_output, df_test_output = current_dataset.divide_data()

        # Evaluate the model
        test_results = nx_graph.test_model(df_test_input, df_test_output)

        return jsonify({
            'status': 'success',
            'message': 'Model evaluation completed',
            'test_loss': float(test_results) if isinstance(test_results, (int, float)) else float(test_results[0])
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)