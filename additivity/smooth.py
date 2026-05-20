from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx

# data
for i in range(1, 127):
    file = str(i)
    cxt = Parser().decode_cxt(f'./data/{file}.cxt')
    lattice = ConceptLattice.from_context(cxt)
    G = lattice.to_networkx()

    tr = nx.transitive_reduction(G)
    tr_smoothed = tr.copy()

    while True:
        nodes_to_remove = [
            node for node in tr_smoothed.nodes() 
            if tr_smoothed.in_degree(node) == 1 and tr_smoothed.out_degree(node) == 1
        ]

        if not nodes_to_remove:
            break

        for node in nodes_to_remove:
            # Get the single predecessor and single successor
            pred = list(tr_smoothed.predecessors(node))[0]
            succ = list(tr_smoothed.successors(node))[0]
            
            # Connect the two outer nodes directly
            tr_smoothed.add_edge(pred, succ)
            
            # Remove the middle node
            tr_smoothed.remove_node(node)
        
        tr_smoothed = nx.transitive_reduction(tr_smoothed)

    import matplotlib.pyplot as plt

    # 1. Use NetworkX + Graphviz (pydot) to calculate hierarchical 'dot' layout coordinates
    # This automatically puts the top elements up high and lower elements down below.
    try:
        pos = nx.drawing.nx_pydot.graphviz_layout(tr_smoothed, prog='dot')
    except (ImportError, OSError):
        # Fallback option if Graphviz/pydot isn't configured on your machine
        # multipartite_layout sets layers based on topological sorting
        for i, layer in enumerate(nx.topological_generations(tr_smoothed)):
            for node in layer:
                tr_smoothed.nodes[node]['layer'] = i
        pos = nx.multipartite_layout(tr_smoothed, subset_key='layer')

    # 2. Draw the layout
    fig, ax = plt.subplots(figsize=(12, 9))

    nx.draw(
        tr_smoothed, 
        pos, 
        ax=ax,
        with_labels=True,        # Displays the remaining Node numbers/IDs
        node_color='#A0CBE2',    # Soft blue
        edge_color='#BBBBBB',    # Clean gray lines
        node_size=800,           
        font_size=10, 
        font_weight='bold',
        arrowsize=15             # Visually shows the direction of the reduction
    )
    print(tr_smoothed.edges)

    plt.title(f"Smoothed Transitive Reduction ({file})", fontsize=14, pad=20)
    plt.show()