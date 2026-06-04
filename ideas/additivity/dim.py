from data import Parser
from linear_extension import SatRealizer

parser = Parser()
for i in range(2, 127):
    cxt = parser.decode_cxt(f'./data/{i}.cxt')
    sat_realizer = SatRealizer(cxt)
    dim, realizer = sat_realizer.realizer()
    print(i, dim)