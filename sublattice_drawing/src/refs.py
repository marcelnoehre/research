import networkx as nx

S7_ref = nx.DiGraph()
S7_ref.add_edges_from([
    ('top', 'a'), ('top', 'b'),  ('top', 'c'),
    ('b', 'd'), ('b', 'e'),
    ('a', 'd'), ('c', 'e'), 
    ('d', 'bot'), ('e', 'bot')
])

B3_ref = nx.DiGraph()
B3_ref.add_edges_from([
    ('bot', 'a'), ('bot', 'b'),  ('bot', 'c'),
    ('a',  'ab'), ('a',  'ac'),
    ('b',  'ab'), ('b',  'bc'),
    ('c',  'ac'), ('c',  'bc'),
    ('ab', 'top'),('ac', 'top'), ('bc', 'top'),
])