import math
import pandas as pd
import networkx as nx
from collections import deque
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def decode_cxt(cxt: str) -> FormalContext:
        '''
        Decode a Burmeister (B) string into a Formal Context.

        The string starts with a B, followed by the dimension of the context and the incidence matrix.

        'x' or 'X' indicates that a object (row) has a feature (column), while a any other character
        indicates that a object does not have a feature. 

        Parameters
        ----------
        cxt : str
            A string representing the burmeister format or a path to the .cxt file

        Returns
        -------
        formal_context : FormalContext
            The formal context.
        '''
        if cxt.endswith('.cxt'):
            with open(cxt, 'r') as f:
                cxt = f.read()

        _, ns, cxt = cxt.split('\n\n')
        n_objs, n_attrs = [int(x) for x in ns.split('\n')]

        cxt = cxt.strip().split('\n')
        obj_names, cxt = cxt[:n_objs], cxt[n_objs:]
        attr_names, cxt = cxt[:n_attrs], cxt[n_attrs:]
        cxt = [[(c == 'X' or c == 'x') for c in line] for line in cxt]

        return FormalContext(data=cxt, object_names=obj_names, attribute_names=attr_names)

def _lectically_smaller(attributes: list, intent_a: set, intent_b: set) -> bool:
    '''
    Check if concept A is lectically smaller than concept B.
    
    A <L B iff there exists an attribute m in M such that:
        - m is the smallest attribute in (intent_b - intent_a) \cup (intent_a - intent_b)
        - m \in intent_b (B contains it, A does not)

    Parameters
    ----------
    intent_a : set
        Intent of concept A
    intent_b : set
        Intent of concept B

    Returns
    -------
    bool
        True if A is lectically smaller than B
    '''
    if intent_a == intent_b:
        return False
    
    for m in attributes:
        if m in intent_a and m in intent_b:
            continue
        if m not in intent_a and m not in intent_b:
            continue
        # m is the smallest differing element
        # A <L B iff A does NOT contain m
        return m not in intent_a
    
    return False

def compute_lectic_order(concepts: list, intents: dict, attributes: list) -> list:
    '''
    Sort all concepts by the lectic order on their intents.

    Returns
    -------
    list
        Concept IDs sorted lectically
    '''
    
    for i in range(1, len(concepts)):
        key = concepts[i]
        key_intent = intents[key]
        j = i - 1
        while j >= 0 and _lectically_smaller(attributes, key_intent, intents[concepts[j]]):
            concepts[j + 1] = concepts[j]
            j -= 1
        concepts[j + 1] = key

    return concepts

def all_intents(
        lattice: ConceptLattice
    ) -> dict:
    '''
    Compute the intents for all concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice

    Returns
    -------
    intents: Dict[int, Set[str]]
        A dictionary mapping concept IDs to their full intents
    '''
    seen = set({})
    queue = deque({0})
    intents = dict({})

    while queue:
        concept = queue.popleft()
        
        # all parents processed?
        if lattice.parents(concept) <= intents.keys():
            # B_new \cup B_parents
            intents[concept] = lattice.get_concept_new_intent(concept).union(
                *(intents[p] for p in lattice.parents(concept))
            )

            # add children to queue
            seen.update(lattice.children(concept) - intents.keys())
            queue.extend(lattice.children(concept) - intents.keys())

        # readd to queue
        else:
            queue.append(concept)

    return intents

metrics_df = pd.read_csv('metrics.csv')
versions = ['hand_drawn', 'sup_inf_attribute', 'sup_inf_doubly', 'dim_draw', 'dim_flux']
labels = {
    'hand_drawn': 'Hand-Drawn',
    'sup_inf_attribute': 'Attr.-Additive FDP (Zschalig)',
    'sup_inf_doubly': 'Doubly-Additive FDP',
    'dim_draw': 'DimDraw (Dürrschnabel)',
    'dim_flux': 'DimFlux',
    'forum_romanum': 'Forum Romanum',
    'living_beings_and_water': 'Living Beings and Water',
    'drive_concepts': 'Drive Concepts',
    'triangles': 'Properties of Triangles',
    'convex_ordinal': 'Convex-Ordinal Scale'
}
labels_short = {
    'hand_drawn': 'Hand-Drawn',
    'sup_inf_attribute': 'Attr.-Additive',
    'sup_inf_doubly': 'Doubly-Additive',
    'dim_draw': 'DimDraw',
    'dim_flux': 'DimFlux',
    'forum_romanum': r'\makecell[c]{Forum \\ Romanum}',
    'living_beings_and_water': r'\makecell[c]{Living Beings \\ and Water}',
    'drive_concepts': r'\makecell[c]{Drive \\ Concepts}',
    'triangles': r'\makecell[c]{Properties \\ of Triangles}',
    'convex_ordinal': r'\makecell[c]{Convex-Ordinal \\ Scale}'
}
metric_labels = {
    'AR': 'Angular Resolution',
    'Asp': 'Aspect Ratio',
    'CA': 'Crossing Angle',
    'EC': 'Edge Crossings',
    'EL': 'Edge Length Deviation',
    'EO': 'Edge Orthogonality',
    'KSM': 'Kruskal Stress Metric',
    'NP': 'Neighbourhood Preservation',
    'NEO': 'Node Edge Occlusion',
    'NR': 'Node Resolution',
    'NU': 'Node Uniformity'
}
headers = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 97, 103, 109, 115, 121]
latex_code = [
    r'\documentclass{article}',
    r'\input{macros}',
    r'\begin{document}',
    r'\input{title}',
    r'\input{introduction}',
    r'\section{All lattices with four meet-irreducibles}\label{sec:m4}'
]

for _cxt in range(1, 127):
    latex_code.append(str(_cxt))
    for _version in versions:
        _metrics = metrics_df[(metrics_df['lattice'] == str(_cxt)) & (metrics_df['version'] == _version)]
        
        _path_cxt = f'contexts/m4/{_cxt}.cxt'
        _path_pos = f'positions/{_version}/{_cxt}.pos'
        
        _context = decode_cxt(_path_cxt)
        _lattice = ConceptLattice.from_context(_context)
        _concepts = list(_lattice.to_networkx().nodes)
        _intents = all_intents(_lattice)
        _attributes = _context.attribute_names
        _lectic_order = compute_lectic_order(_concepts, _intents, _attributes)
        
        with open(_path_pos, 'r') as f:
            coords = [tuple(map(float, line.split()[:2])) for line in f if line.strip()]
        _positions = [coords[_lectic_order.index(c)] for c in _concepts]

        assert len(_concepts) == len(_positions)
        
        latex_code.append(r'\begin{minipage}[c]{0.18\textwidth}')
        latex_code.append(r'  \centering')
        if _cxt in headers:
            latex_code.append(fr'  \textbf{{{labels[_version]}}}\\[2ex]' + (r'~\\' if _version in ['hand_drawn', 'dim_draw_double'] else ''))
        
        latex_code.append(r'  \begin{adjustbox}{max width=\textwidth}')
        latex_code.append(r'    \begin{tikzpicture}[scale=.8]')
        latex_code.append(r'      \begin{scope}[every node/.style={circle, thick, draw, fill=white, inner sep=0pt, minimum size=1.5mm}]')
        
        for c in _concepts:
            latex_code.append(fr'      \node ({c}) at ({_positions[c][0]}, {_positions[c][1]}) {{}};')

        latex_code.append(r'      \end{scope}')

        for (i, j) in set(nx.transitive_reduction(_lattice.to_networkx()).edges):
            latex_code.append(fr'      \draw ({i}) -- ({j});')

        latex_code.append(r'    \end{tikzpicture}')
        latex_code.append(r'  \end{adjustbox}')

        latex_code.append(r'\\[1ex]')


        latex_code.append(fr'\scriptsize {(_metrics['energy'].iloc[0] / 80):.2f}' + (r'' if _metrics['additive'].iloc[0] else r'$^{\ast}$'))

        if _version == 'hand_drawn':
            latex_code.append(r'\\ ~ \\')
        else:
            latex_code.append(fr'\\ \scriptsize \textit{{{_metrics['hand'].iloc[0]}}}')

        if _version == 'dim_flux':
            latex_code.append(r'\end{minipage} \\[4ex]')
        else:
            latex_code.append(r'\end{minipage} \hfill')

latex_code.append(r'\clearpage')
latex_code.append(r'\section{Real-world examples}\label{sec:real-world}')

for _cxt in ['forum_romanum', 'living_beings_and_water', 'drive_concepts', 'triangles', 'convex_ordinal']:
    latex_code.append(fr'\subsection{{{labels[_cxt]}}}\label{{subsec:{_cxt}}}')
    for _version in versions:
        _metrics = metrics_df[(metrics_df['lattice'] == str(_cxt)) & (metrics_df['version'] == _version)]
        
        _path_cxt = f'contexts/real-world/{_cxt}.cxt'
        _path_pos = f'positions/{_version}/{_cxt}.pos'
        
        _context = decode_cxt(_path_cxt)
        _lattice = ConceptLattice.from_context(_context)
        _concepts = list(_lattice.to_networkx().nodes)
        _intents = all_intents(_lattice)
        _attributes = _context.attribute_names
        _lectic_order = compute_lectic_order(_concepts, _intents, _attributes)
        
        with open(_path_pos, 'r') as f:
            coords = [tuple(map(float, line.split()[:2])) for line in f if line.strip()]
        _positions = [coords[_lectic_order.index(c)] for c in _concepts]

        assert len(_concepts) == len(_positions)

        latex_code.append(rf'\begin{{minipage}}[c]{{{1.0 if _version == 'hand_drawn' else 0.5}\textwidth}}')
        latex_code.append(r'  \centering')
        latex_code.append(fr'  \textbf{{{labels[_version]}}}\\[1ex]' + (r'~\\' if _version == 'hand_drawn' else ''))
        latex_code.append(r'  \begin{adjustbox}{max width=\textwidth}')
        latex_code.append(r'    \begin{tikzpicture}[scale=.95]')
        latex_code.append(r'      \begin{scope}[every node/.style={circle, thick, draw, fill=white, inner sep=0pt, minimum size=1.5mm}]')
        
        for c in _concepts:
            latex_code.append(fr'      \node ({c}) at ({_positions[c][0]}, {_positions[c][1]}) {{}};')

        latex_code.append(r'      \end{scope}')

        for (i, j) in set(nx.transitive_reduction(_lattice.to_networkx()).edges):
            latex_code.append(fr'      \draw ({i}) -- ({j});')

        latex_code.append(r'    \end{tikzpicture}')
        latex_code.append(r'  \end{adjustbox}')

        latex_code.append(r'\\[1ex]')
        latex_code.append(fr'\scriptsize {(_metrics['energy'].iloc[0] / 80):.2f}' + (r'' if _metrics['additive'].iloc[0] else r'$^{\ast}$'))
        
        if _version != 'hand_drawn':
            latex_code.append(fr'\\ \scriptsize \textit{{{_metrics['hand'].iloc[0]}}}')

        if _version == 'dim_flux':
            latex_code.append(r'\end{minipage} \\[4ex]')
        else:
            latex_code.append(r'\end{minipage} \hfill')

        # latex_code.append(r'\end{minipage}%')

        if _version in ['hand_drawn', 'sup_inf_doubly']:
            latex_code.extend(['', r'\vspace{4em}', ''])

    latex_code.append(r'\clearpage')

latex_code.append(r'\input{metrics}')

metrics = ['AR','Asp','CA','EC','EL','EO','KSM','NP','NEO','NR','NU']
lattices = [str(i) for i in range(1, 64)] + ['forum_romanum', 'living_beings_and_water', 'drive_concepts'] + [str(i) for i in range(64, 127)] + ['triangles', 'convex_ordinal', None]

for _metric in metrics:
    _mid = math.ceil(len(lattices) / 2)
    _left = lattices[:_mid]
    _right = lattices[_mid:]
    _n_versions = len(versions)
    _version_header = ' & '.join([r'\rotatebox{90}{' + labels_short[_version] + r'}' for _version in versions])
    header = (r'Lattice & ' + _version_header + r' & Lattice & ' + _version_header + r' \\ \midrule')

    latex_code.append(r'\clearpage')
    latex_code.append(fr'\subsection{{{metric_labels[_metric]}}}\label{{subsec:{_metric}}}')
    latex_code.append(r'{\normalsize')
    latex_code.append(r'\begin{longtable}{' + ('c' + 'c' * _n_versions + '@{\\qquad}c' + 'c' * _n_versions) + '}')
    latex_code.append(r'\toprule')
    latex_code.append(header)
    latex_code.append(r'\endfirsthead')
    latex_code.append(r'\toprule')
    latex_code.append(header)
    latex_code.append(r'\endhead')
    latex_code.append(r'\midrule \multicolumn{' + str((_n_versions + 1) * 2) + r'}{r}{\textit{Continued...}} \\ \endfoot')
    latex_code.append(r'\bottomrule \endlastfoot')

    def _metric_val(lat):
        if lat is None:
            return [('-', None)] * _n_versions
        
        vals = []
        for _version in versions:
            _metrics = metrics_df[(metrics_df['lattice'] == lat) & (metrics_df['version'] == _version)]
            _val = f'{_metrics[_metric].iloc[0]:.2f}'
            vals.append((_val, float(_val)))

        return vals

    def _bold(s, v, best):
        return r'\textbf{' + s + r'}' if v is not None and v == best else s

    for _left_lat, _right_lat in zip(_left, _right):
        _vals_left  = _metric_val(_left_lat)
        _vals_right = _metric_val(_right_lat)

        def _best(vals_raw):
            numeric = [v for _, v in vals_raw if v is not None]
            return min(numeric, key=lambda x: abs(x - 1.0)) if numeric else None

        latex_code.append((
            f'{labels_short.get(_left_lat, _left_lat)} & ' + ' & '.join([_bold(s, v, _best(_vals_left))  for s, v in _vals_left]) +
            f' & {labels_short.get(_right_lat, _right_lat)} & ' + ' & '.join([_bold(s, v, _best(_vals_right)) for s, v in _vals_right]) +
            r' \\'
        ))

    latex_code.append(r'\caption{Metric: ' + metric_labels[_metric] + r'} \\')
    latex_code.append(r'\end{longtable}')
    latex_code.append(r'}')

latex_code.append(r'\end{document}')

with open('dim_flux_comparison.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(latex_code))