from pyvis import network as net
import networkx as nx
import tempfile
import webbrowser
import os

# Crear una red con PyVis
g = net.Network(notebook=False, cdn_resources="remote")
nxg = nx.Graph()


# Función para añadir nodos y aristas desde Python
def add_node(node_id, label):
    nxg.add_node(node_id, label=label)


def add_edge(source, target):
    nxg.add_edge(source, target)


def remove_node(node_id):
    if node_id in nxg:
        nxg.remove_node(node_id)


def remove_edge(source, target):
    if nxg.has_edge(source, target):
        nxg.remove_edge(source, target)


# Función para actualizar el HTML de la red con el menú de control
def actualizar_grafo():
    g.from_nx(nxg)
    nxg.nodes
    nxg.edges
    # Crear archivo HTML temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    g.write_html(temp_file.name)

    # HTML adicional para los controles de nodos/aristas en el menú
    custom_html = """
        <div style="padding: 10px; font-family: Arial;">
            <h3>Control de Red</h3>
            <label>ID Nodo:</label><input type="number" id="node-id" placeholder="ID Nodo">
            <label>Etiqueta:</label><input type="text" id="node-label" placeholder="Etiqueta">
            <button onclick="addNode()">Añadir Nodo</button>
            <br><br>
            <label>Nodo Origen:</label><input type="number" id="edge-source" placeholder="Origen">
            <label>Nodo Destino:</label><input type="number" id="edge-target" placeholder="Destino">
            <button onclick="addEdge()">Añadir Arista</button>
            <br><br>
            <label>ID Nodo a eliminar:</label><input type="number" id="remove-node-id" placeholder="ID Nodo">
            <button onclick="removeNode()">Eliminar Nodo</button>
            <br><br>
            <label>Arista a eliminar (Origen - Destino):</label>
            <input type="number" id="remove-edge-source" placeholder="Origen">
            <input type="number" id="remove-edge-target" placeholder="Destino">
            <button onclick="removeEdge()">Eliminar Arista</button>
        </div>

        <script type="text/javascript">
            function addNode() {
                var nodeId = document.getElementById('node-id').value;
                var label = document.getElementById('node-label').value;
                network.body.data.nodes.add({id: parseInt(nodeId), label: label});
            }

            function addEdge() {
                var source = document.getElementById('edge-source').value;
                var target = document.getElementById('edge-target').value;
                network.body.data.edges.add({from: parseInt(source), to: parseInt(target)});
            }

            function removeNode() {
                var nodeId = document.getElementById('remove-node-id').value;
                network.body.data.nodes.remove({id: parseInt(nodeId)});
            }

            function removeEdge() {
                var source = document.getElementById('remove-edge-source').value;
                var target = document.getElementById('remove-edge-target').value;
                var edgeId = network.body.data.edges.getIds({filter: function(edge) {
                    return edge.from === parseInt(source) && edge.to === parseInt(target);
                }})[0];
                if (edgeId !== undefined) {
                    network.body.data.edges.remove(edgeId);
                }
            }
        </script>
    """

    # Leer el contenido del archivo HTML generado
    with open(temp_file.name, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Insertar el HTML adicional antes de la etiqueta </body>
    updated_html = html_content.replace("</body>", custom_html + "</body>")

    # Sobrescribir el archivo con el contenido actualizado
    with open(temp_file.name, "w", encoding="utf-8") as file:
        file.write(updated_html)

    # Abrir el archivo HTML actualizado en el navegador
    webbrowser.open('file://' + os.path.abspath(temp_file.name))


# Generar el grafo inicial
actualizar_grafo()
