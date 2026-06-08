import os
import matplotlib.pyplot as plt
import networkx as nx


def draw_graphml_with_stored_positions(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        print(f"Reading graph from {file_path}...")
        G = nx.read_graphml(file_path)

        # 1. Extract stored positions from node attributes
        pos = {}
        missing_coords = False

        for node, data in G.nodes(data=True):
            # GraphML often stores coordinates as 'x' and 'y' (or sometimes 'graphics_x', etc.)
            # We cast them to float just in case they were read as strings
            if "x" in data and "y" in data:
                pos[node] = (float(data["x"]), float(data["y"]))
            else:
                missing_coords = True

        # 2. Fallback if positions aren't found or are incomplete
        if missing_coords or not pos:
            print(
                "Warning: Some or all nodes are missing 'x' and 'y' attributes."
            )
            print("Falling back to spring layout.")
            pos = nx.spring_layout(G, seed=42)
        else:
            print("Successfully loaded stored positions from file.")

        # 3. Draw the graph
        plt.figure(figsize=(10, 8))
        plt.title(f"Graph Visualization: {os.path.basename(file_path)}")

        nx.draw_networkx_nodes(G, pos, node_size=600, node_color="skyblue")
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color="gray")
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

        plt.axis("off")
        plt.tight_layout()
        print("Displaying window...")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    graphml_path = "../lattice_metrics/graphs/hand_drawn/58.graphml"
    draw_graphml_with_stored_positions(graphml_path)