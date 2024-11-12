from pyvis import network as net
import networkx as nx
from IPython.display import IFrame, display, HTML
import ipywidgets as widgets
import tempfile

# Crear una red con PyVis
g = net.Network(notebook=True, cdn_resources="remote")
nxg = nx.Graph()


# Crear una función para actualizar el gráfico
def actualizar_grafo():
    g.from_nx(nxg)
    g.show("example.html")
    # Leer el archivo HTML y mostrarlo en Colab
    with open("example.html", "r") as file:
       display(HTML(file.read()))


# Creamos una red neuronal de presentación
# Capa de entrada (en un futuro el número de neurnas en la capa de entrada se meterán en función del dataset )
nxg.add_node(0, label="0")
# Capa de oculta
for x in range(1,6):
    nxg.add_node(x, label="x")
    nxg.add_edge(0, x)
    nxg.add_edge(x, 7)
# Capa de salida
nxg.add_node(7, label="7")

actualizar_grafo()


# Funciones para añadir nodos y aristas
def add_node(b):
    node_id = int(node_id_input.value)
    label = label_input.value
    nxg.add_node(node_id, label=label)
    actualizar_grafo()

def add_edge(b):
    source = int(source_input.value)
    target = int(target_input.value)
    nxg.add_edge(source, target)
    actualizar_grafo()

# Funciones para eliminar nodos y aristas
def remove_node(b):
    node_id = int(node_id_remove_input.value)
    if node_id in nxg:
        nxg.remove_node(node_id)
        actualizar_grafo()

def remove_edge(b):
    source = int(source_remove_input.value)
    target = int(target_remove_input.value)
    if nxg.has_edge(source, target):
        nxg.remove_edge(source, target)
        actualizar_grafo()

# Crear widgets para añadir nodos
node_id_input = widgets.Text(placeholder='ID del Nodo')
label_input = widgets.Text(placeholder='Etiqueta del Nodo')
add_node_button = widgets.Button(description="Añadir Nodo")
add_node_button.on_click(add_node)

# Crear widgets para añadir aristas
source_input = widgets.Text(placeholder='Nodo Origen')
target_input = widgets.Text(placeholder='Nodo Destino')
add_edge_button = widgets.Button(description="Añadir Arista")
add_edge_button.on_click(add_edge)

# Crear widgets para eliminar nodos
node_id_remove_input = widgets.Text(placeholder='ID del Nodo a Eliminar')
remove_node_button = widgets.Button(description="Eliminar Nodo")
remove_node_button.on_click(remove_node)

# Crear widgets para eliminar aristas
source_remove_input = widgets.Text(placeholder='Nodo Origen')
target_remove_input = widgets.Text(placeholder='Nodo Destino')
remove_edge_button = widgets.Button(description="Eliminar Arista")
remove_edge_button.on_click(remove_edge)

# Mostrar los widgets
display(widgets.HBox([node_id_input, label_input, add_node_button]))
display(widgets.HBox([source_input, target_input, add_edge_button]))
display(widgets.HBox([node_id_remove_input, remove_node_button]))
display(widgets.HBox([source_remove_input, target_remove_input, remove_edge_button]))

# Mostrar el grafo inicial vacío
actualizar_grafo()