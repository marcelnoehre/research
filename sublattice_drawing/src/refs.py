import networkx as nx

N5_ref = nx.DiGraph()
N5_ref.add_edges_from([
    ('bot', 'a'), ('bot', 'b'), ('b', 'c'), ('a', 'top'), ('c', 'top')
])

B3_ref = nx.DiGraph()
B3_ref.add_edges_from([
    ('bot', 'a'), ('bot', 'b'),  ('bot', 'c'),
    ('a',  'ab'), ('a',  'ac'),
    ('b',  'ab'), ('b',  'bc'),
    ('c',  'ac'), ('c',  'bc'),
    ('ab', 'top'),('ac', 'top'), ('bc', 'top'),
])