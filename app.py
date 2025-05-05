import os

from flask import Flask, render_template, url_for, request, jsonify

from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX
from src.neuronalNetworks.neuronal_Network_Pyvis import PyVisNeuronalNetwork

app = Flask(__name__)

# Asegurar que el directorio static/graphs existe
os.makedirs("static/graphs", exist_ok=True)

# Crear una instancia de NeuronalNetworkX y PyVisNeuronalNetwork
global nx_graph
global pyvis_graph


# Inicializar la red neuronal si no existe
def initialize_networks():
    global nx_graph, pyvis_graph

    # Comprobar si la red ya ha sido inicializada
    if 'neural_network' not in globals() or nx_graph is None:
        # Valores predeterminados para la red - ajústalos según tus necesidades
        num_inputs = 3
        num_outputs = 1
        nx_graph = NeuronalNetworkX(num_inputs, num_outputs)
        nx_graph.defaultNetwork()  # Inicializa con la red por defecto

    # Inicializar o actualizar la red PyVis
    if 'pyvis_network' not in globals() or pyvis_graph is None:
        pyvis_graph = PyVisNeuronalNetwork()

    pyvis_graph.actualise_graph(nx_graph)
    pyvis_graph.generate_HTML()


# Inicializar las redes al inicio
initialize_networks()

@app.route('/')
def index():
    # Asegurar que el directorio static/graphs existe
    #os.makedirs("static/graphs", exist_ok=True)
    # Generar el grafo antes de renderizar la plantilla
    #nx_graph.defaultNetwork()
    #pyvis_graph.actualise_graph(nx_graph)  # Convierte el grafo de NetworkX a PyVis
    #pyvis_graph.generate_HTML()  # Guarda el HTML en "static/graphs/pyvis.html"

    return render_template("dataset_selection.html")


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