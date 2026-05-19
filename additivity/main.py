import numpy as np
from data import Parser
from fcapy.lattice import ConceptLattice
from latex import latex_export
from lectic import *
import networkx as nx

file = 'drive_concepts'
parser = Parser()
cxt = parser.decode_cxt(f'./data/{file}.cxt')
cxt.attribute_names
lattice = ConceptLattice.from_context(cxt)
concepts = list(lattice.to_networkx().nodes)
intents = all_intents(lattice)
extents = all_extents(lattice)
objects = cxt.object_names
attributes = cxt.attribute_names
M = set(attributes)
G = set(objects)
lectic_order = compute_lectic_order(concepts, intents, attributes)
with open(f'./data/{file}.pos', 'r') as f:
    positions = [tuple(map(float, line.split()[:2])) for line in f if line.strip()]

with open(f'./data/{file}.pos', 'r') as f:
    coords = [tuple(map(float, line.split()[:2])) for line in f if line.strip()]

positions = [coords[c] for c in lectic_order]

srm = np.array([
    [
        1 if e in (extents[c] | (M - intents[c])) else 0
        for e in objects + attributes
    ]
    for c in lectic_order
])

basis = np.zeros((len(srm),0))
for srm_col_n in range(srm.shape[1]):
    projection = np.zeros((len(srm),1))
    srm_col = srm[:,srm_col_n:srm_col_n+1]
    if basis.shape[1] > 0:
        for bcolnum in range(basis.shape[1]):
            bcol = basis[:,bcolnum:bcolnum+1]
            projection += np.dot(srm_col.T,bcol)*bcol
    newcol = srm_col - projection
    norm = np.linalg.norm(newcol)
    if norm > 1.e-05:
        newcol = newcol/norm
        basis = np.column_stack((basis,newcol))

# (N_c, 2) position matrix in lectic order
xy = np.array(positions)
anchor_point = xy[-1] 
xy_translated = xy - anchor_point
# project onto column space of SRM
projected_xy = basis @ (basis.T @ xy_translated)
additive_positions = [projected_xy[i] for i, _ in enumerate(lectic_order)]

print(xy_translated)
print(projected_xy)
are_equal = np.allclose(xy_translated, projected_xy)
print(are_equal)

relations = list(nx.transitive_reduction(lattice.to_networkx()).edges)
relations_lectic = list([(lectic_order.index(a), lectic_order.index(b)) for a, b in relations])

for c in concepts:
    print(c, lectic_order.index(c))

latex_export(xy_translated, relations_lectic, file + 'original')
latex_export(projected_xy, relations_lectic, file + 'additive')