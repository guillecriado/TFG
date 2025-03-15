import os

from flask import Flask, render_template, url_for

from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX
from src.neuronalNetworks.neuronal_Network_Pyvis import PyVisNeuronalNetwork

app = Flask(__name__)

# Asegurar que el directorio static/graphs existe
os.makedirs("src/static/graphs", exist_ok=True)

# Crear una instancia de NeuronalNetworkX y PyVisNeuronalNetwork
nx_graph = NeuronalNetworkX(3,1)  # Aquí asumo que puedes inicializarlo sin argumentos
pyvis_graph = PyVisNeuronalNetwork()



@app.route('/')
def index():
    # Generar el grafo antes de renderizar la plantilla
    nx_graph.defaultNetwork()
    pyvis_graph.actualise_graph(nx_graph)  # Convierte el grafo de NetworkX a PyVis
    pyvis_graph.generate_HTML()  # Guarda el HTML en "static/graphs/pyvis.html"

    return render_template("hola.html")

if __name__ == "__main__":
    app.run(debug=True)