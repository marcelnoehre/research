from data import Parser
from fcapy.lattice import ConceptLattice

parser = Parser()
cxt = parser.decode_cxt('eurovision_support.cxt')
lattice = ConceptLattice.from_context(cxt)

for i in range(16):
    extent = lattice.get_concept_new_extent(i)
    string = ''
    for song in extent:
        string += f"'{song.split('_')[1]}', "
    print(string)