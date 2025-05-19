import os

import pyvis as pv
from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX


class PyVisNeuronalNetwork:
    def __init__(self):
        self.graph : pv.network.Network = None

    def actualise_graph(self, nx: NeuronalNetworkX):
        self.graph = pv.network.Network()
        self.graph.from_nx(nx.nxg)

    def generate_HTML(self, nx: NeuronalNetworkX):
        # Asegurarse de limpiar cualquier contenido antiguo
        self.graph = pv.network.Network()
        self.graph.from_nx(nx.nxg)
        self.graph.save_graph("static/graphs/pyvis.html")


