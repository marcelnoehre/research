from data.parser import Parser
from fca.lattice import Lattice

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
FILE = 'living_beings_and_water_original'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{FILE}.cxt')
print('Formal Context')
print(cxt.print_data())

with open(f'../data/{FILE}.pos', 'r') as f:
    coords = {
        c: tuple(map(float, line.split()[:2]))
        for c, line in enumerate(f) if line.strip()
    }

lattice = Lattice(cxt)
edges_list = list(lattice.cover_relations())