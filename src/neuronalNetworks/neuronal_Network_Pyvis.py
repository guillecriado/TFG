import os

import pyvis as pv
from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX


class PyVisNeuronalNetwork:
    def __init__(self):
        self.graph : pv.network.Network = None

    def actualise_graph(self, nx: NeuronalNetworkX):
        # Create a new network with specific settings for better visualization
        self.graph = pv.network.Network(
            height="500px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )

        # Configure physics for better layout
        self.graph.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "solver": "hierarchicalRepulsion",
                "hierarchicalRepulsion": {
                    "centralGravity": 0.2,
                    "springLength": 100,
                    "springConstant": 0.01,
                    "nodeDistance": 150,
                    "damping": 0.09
                }
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 200
                }
            },
            "nodes": {
                "shape": "circle",
                "font": {
                    "size": 12,
                    "color": "black"
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "smooth": {
                    "type": "continuous"
                }
            }
        }
        """)

        # Import the networkx graph
        self.graph.from_nx(nx.nxg)

    def generate_HTML(self, nx: NeuronalNetworkX):
        # Ensure the directory exists
        os.makedirs("static/graphs", exist_ok=True)

        # First actualise the graph to ensure it's up to date
        self.actualise_graph(nx)

        # Generate the HTML file
        html_path = "static/graphs/pyvis.html"

        # Save the graph
        self.graph.save_graph(html_path)

        # Read the generated HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Add meta tags to prevent caching
        no_cache_meta = """
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        """

        # Insert the meta tags after the <head> tag
        html_content = html_content.replace('<head>', '<head>\n' + no_cache_meta)

        # Add custom CSS for better visualization
        custom_css = """
        <style>
            body {
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
            #mynetwork {
                border: none;
            }
        </style>
        """

        # Insert custom CSS before </head>
        html_content = html_content.replace('</head>', custom_css + '\n</head>')

        # Write the modified HTML back
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


