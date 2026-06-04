from src.utils.visualize import *
from src.utils.variables import Variables
from src.dim_flux.realizer import Realizer
from src.fdp.sup_inf import SupInfGraph
from src.fdp.init_layout import InitLayout
from src.fdp.forces import ForceDirectedPlacement
from src.dim_flux.snap_to_grid import SnapToGrid

def main():
    # for name in ['Forum-Romanum', 'living_beings_and_water', 'car', 'triangles']: #, 'convex-ordinal']:
    #     cxt = f'evaluation/data/study_reduced/{name}.cxt'
    #     vars = Variables(name, cxt, {
    for i in range(1, 127):
        cxt = f'evaluation/data/m4_dim_draw/{i}.cxt'
        vars = Variables(str(i), cxt, {
            'plot_si_graph': False,
            'si_graph_annotations':  False,
            'plot_initial_layout':  False,
            'initial_layout_annotations':  False,
            'plot_optimized_layout':  False,
            'optimized_layout_annotations':  False,
            'plot_individual_forces':  False,
            'plot_combined_forces':  False,
            'plot_gradients':  False,
            'plot_origin':  False
        })

        mode = 'DimFlux' # 'PlanarityEnhancer'

        if mode == 'DimFlux':
            realizer = Realizer(vars)
            vars.base_vectors = realizer.base_vectors
            vars.coordinates = realizer.coordinates

            if vars.args.plot_initial_layout:
                plot_lattice(vars, 'Initial layout (Projected DimDraw)', vars.args.initial_layout_annotations, False)

        else:
            # Sup-Inf Graph
            sup_inf_graph = SupInfGraph(vars)
            vars.scalars = sup_inf_graph.scalars
            vars.order = sup_inf_graph.order
            vars.d_si_points = sup_inf_graph.d_si_points
            vars.n_1 = sup_inf_graph.n_1
            vars.n_2 = sup_inf_graph.n_2

            if vars.args.plot_si_graph:
                plot_si_graph(vars)

            # Initial Layout
            initial_layout = InitLayout(vars)
            vars.base_vectors = initial_layout.base_vectors
            vars.coordinates = initial_layout.coordinates

            if vars.args.plot_initial_layout:
                plot_lattice(vars, 'Initial layout', vars.args.initial_layout_annotations, False)

        # Optimize Layout
        forces = ForceDirectedPlacement(vars)
        vars.base_vectors = forces.base_vectors
        vars.coordinates = forces.coordinates
        vars.final_forces = forces.final_forces

        if vars.args.plot_optimized_layout:
            plot_lattice(
                vars, 'Optimized layout', vars.args.optimized_layout_annotations,
                vars.args.plot_individual_forces or vars.args.plot_combined_forces or vars.args.plot_gradients
            )
        
        snapped = SnapToGrid(vars)
        vars.base_vectors = snapped.base_vectors
        vars.coordinates = snapped.coordinates

        # plot_lattice(
        #     vars, 'Final layout', vars.args.optimized_layout_annotations,
        #     vars.args.plot_individual_forces or vars.args.plot_combined_forces or vars.args.plot_gradients
        # )

        pos = [f'{vars.coordinates[c][0]} {vars.coordinates[c][1]}' for c in vars.concepts]
        with open(f'evaluation/positions/dim_draw_double/{vars.name}.pos', 'w', encoding='utf-8') as f:
            f.write('\n'.join(pos))

if __name__ == "__main__":
    main()
