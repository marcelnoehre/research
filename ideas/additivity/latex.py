def latex_export(positions, relations, file_name: str):
    '''
    Generate a LaTeX PGF/TikZ representation of the concept lattice.

    Parameters
    ----------
    vars : Variables
        The container holding the lattice and coordinates
    prefix : str
        The subdirectory prefix for the output file
    '''
    lines = [
        r'\begin{tikzpicture}[scale=0.6]',
        r'  \begin{scope}[every node/.style={circle, thick, draw, fill=white, inner sep=0pt, minimum size=2mm}]'
    ]

    # vertices
    for c in range(len(positions)):
        (x, y) = positions[c]
        lines.append(fr'    \node ({c}) at ({x:.3f}, {y:.3f}) {{}};')

    lines.append(r'  \end{scope}')

    # edges
    for (i, j) in relations:
        lines.append(fr'  \draw[thick] ({i}) -- ({j});')
        
    lines.append(r'\end{tikzpicture}')

    with open(f'data/{file_name}.tex', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))