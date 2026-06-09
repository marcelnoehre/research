from data import Parser
from fcapy.lattice import ConceptLattice
from src.realizer import SatRealizer
from src.cut import *
from src.refs import *
from src.find_sublattices import *
from src.positions import *

file = '8'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
lat = ConceptLattice.from_context(cxt)
G = nx.transitive_reduction(lat.to_networkx())

anchors, parts = parts_as_lattices(G, lat)

positions = {}
previous_cut = (0, 0)

print(f"anchors (top -> bottom): {anchors}\n")
print(f"{len(parts)} part(s):")
for a, b, region, sub_lat, index_map in reversed(parts):
    inner = len(region) - 2
    print(f"A = {a}  ->  B = {b}   ({inner} node(s) enclosed)")
    print(f"    region:    {sorted(region, key=str)}")
    print(f"    index_map: {index_map}")
    print(f"    sub_lat:   {len(sub_lat)} concepts")

    sat_realizer = SatRealizer(sub_lat)
    
    dim, realizer = sat_realizer.realizer()
    print(realizer)
    ranks = {}

    if dim == 1:
        new_positions = total_ordering(realizer)
    elif dim == 2:
        s7 = find_sublattice(sub_lat.to_networkx(), S7_ref)
        if s7:
            new_positions = S7_positions(s7, realizer)

        print(all_maximal_chains(sub_lat.to_networkx()))
        print(parallel_intervals(sub_lat.to_networkx()))

        
    elif dim == 3:
        find_sublattice(sub_lat.to_networkx(), B3_ref)
    else:
        raise ValueError('Not implemented')

    new_positions = {index_map[n]: (x + previous_cut[0], y + previous_cut[1]) for n, (x, y) in new_positions.items()}
    previous_cut = max(new_positions.values(), key=lambda p: p[1])
    for n, pos in new_positions.items():
        positions[n] = pos

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_aspect('equal')
nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', arrows=True, ax=ax)
plt.show()