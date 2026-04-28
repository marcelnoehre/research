import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from src.utils.variables import Variables
from src.fca.lattice import cover_relations

def plot_si_graph(vars: Variables):
    '''
    Plot the Supremum-Infimum graph representing the relationship 
    between objects and attributes in the layout space.

    Parameters
    ----------
    vars : Variables
        The container class holding the data
    '''
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title(f'Sup Inf Graph: {vars.cxt}')

    # objects and attributes as vertices
    for i, (x, y) in enumerate(vars.d_si_points):
        plt.scatter(x, y, facecolor='white', edgecolor='black', linewidth=2.5, s=150, zorder=4)
        if vars.args.si_graph_annotations:
            plt.annotate(
                vars.elements[i],
                (x, y),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

    # edges approximate d_SI
    for i, j in combinations(range(vars.N_e), 2):
        x_0, y_0 = np.array(vars.d_si_points[i])
        x_1, y_1 = np.array(vars.d_si_points[j])
        plt.plot([x_0, x_1], [y_0, y_1], color='black', linewidth=1, zorder=2)

    # f_max
    plt.plot([vars.n_1[0], vars.n_2[0]], [vars.n_1[1], vars.n_2[1]], color='black', linewidth=4, zorder=2)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_lattice(vars: Variables, title: str, annotations: bool, forces: bool):
    '''
    Plot the Concept Lattice and optionally visualize the force vectors.

    Parameters
    ----------
    vars : Variables
        The container class holding the lattice and coordinate mapping
    title : str
        The title for the plot window
    annotations : bool
        Whether to display object and attribute labels
    forces : bool
        Whether to overlay force vectors on the concepts
    '''
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title(title)

    if vars.args.plot_origin:
        plt.scatter(0, 0, facecolor='red', edgecolor='pink', linewidth=5, s=150, zorder=10)
    
    # vertices
    for concept in vars.concepts:
        (x, y) = vars.coordinates[concept]
        plt.scatter(x, y, facecolor='white', edgecolor='black', linewidth=2.5, s=150, zorder=4)
        if annotations:
            # attributes
            plt.annotate(
                ','.join(vars.lattice.get_concept_new_intent(concept)),
                (x, y),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
            # objects
            plt.annotate(
                ','.join(vars.lattice.get_concept_new_extent(concept)),
                (x, y),
                textcoords='offset points',
                xytext=(0, -10),
                ha='center',
                va='top',
                fontsize=12,
                fontweight='bold'
            )

    # edges
    for (i, j) in cover_relations(vars.lattice):
        x_0, y_0 = np.array(vars.coordinates[i])
        x_1, y_1 = np.array(vars.coordinates[j])
        plt.plot([x_0, x_1], [y_0, y_1], color='black', linewidth=2.5, zorder=2)

    if forces:
        _plot_individual_forces(vars)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def latex_export(vars: Variables, prefix: str):
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
    for c in vars.concepts:
        (x, y) = vars.coordinates[c]
        lines.append(fr'    \node ({c}) at ({x:.3f}, {y:.3f}) {{}};')

    lines.append(r'  \end{scope}')

    # edges
    for (i, j) in cover_relations(vars.lattice):
        lines.append(fr'  \draw[thick] ({i}) -- ({j});')
        
    lines.append(r'\end{tikzpicture}')

    with open(f'results_{prefix}/{vars.cxt}.tex', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def pdf_export(vars: Variables, prefix: str):
    '''
    Export the Concept Lattice as a PDF file.

    Parameters
    ----------
    vars : Variables
        The container holding the lattice and coordinates
    prefix : str
        The subdirectory prefix for the output file
    '''
    plt.figure(figsize=(6, 6))

    # vertices
    for concept in vars.concepts:
        (x, y) = vars.coordinates[concept]
        plt.scatter(x, y, facecolor='white', edgecolor='black', linewidth=3.5, s=350, zorder=4)

    # edges
    for (i, j) in cover_relations(vars.lattice):
        x_0, y_0 = np.array(vars.coordinates[i])
        x_1, y_1 = np.array(vars.coordinates[j])
        plt.plot([x_0, x_1], [y_0, y_1], color='black', linewidth=3.5, zorder=2)

    file = f'results_{prefix}/{vars.cxt}.pdf'
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file, format='pdf', bbox_inches='tight')
    plt.close()

def _plot_individual_forces(vars: Variables):
    '''
    Internal helper to draw force arrows (repulsive, attractive, gravitational) for each concept.

    Parameters
    ----------
    vars : Variables
        The container class holding coordinates and computed force vectors
    '''
    #configure forces
    if vars.args.plot_individual_forces:
        configs = [
            (['G_rep'],  '#8b0000'),
            (['G_att'],  '#00008b'),
            (['G_grav'], '#006400'),
            (['M_rep'],  '#ff4500'),
            (['M_att'],  '#1e90ff'),
            (['M_grav'], '#32cd32')
        ]
    elif vars.args.plot_combined_forces:
        configs = [
            (['G_rep', 'M_rep'],  'red'),
            (['G_att', 'M_att'],  'blue'),
            (['G_grav', 'M_grav'], 'green'),   
        ]
    elif vars.args.plot_gradients:
        configs = [(['G_rep', 'M_rep', 'G_att', 'M_att', 'G_grav', 'M_grav'], 'orange')]
    else:
        configs = []

    # plot forces for each concept
    for c, (x_c, y_c) in enumerate([vars.coordinates[c] for c in vars.concepts]):
        for forces, color in configs:
            x_force, y_force = np.array([x_c, y_c]) + np.sum([vars.final_forces[f][c] for f in forces], axis=0) * 0.1
            plt.plot([x_c, x_force], [y_c, y_force], color=color, linewidth=2.5, zorder=2)
            plt.annotate('', 
            xy = (x_force, y_force),
            xytext = (x_c, y_c),
            arrowprops = dict(
                arrowstyle='->', 
                color=color, 
                lw=2.5, 
                mutation_scale=30,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=2)