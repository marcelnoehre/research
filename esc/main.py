from odis import FormalContext

import networkx as nx
from data import Parser
from fcapy.lattice import ConceptLattice

parser = Parser()
cxt = parser.decode_cxt(f'eurovision_binary_context.cxt')
lattice = ConceptLattice.from_context(cxt)
G = lattice.to_networkx()
print('Formal Concepts:', len(G.nodes))
print('edges:', len(nx.transitive_reduction(G).edges))

ctx = FormalContext.from_file('eurovision_binary_context.cxt')
svg = ctx.draw_svg("sugiyama", width=800, height=600)
with open("lattice.svg", "w") as f:
    f.write(svg)