from odis import FormalContext

for i in range(14):
    ctx = FormalContext.from_file(f'cluster_{i}.cxt')
    svg = ctx.draw_svg("dimdraw", width=800, height=600)
    with open(f'cluster_{i}.svg', "w") as f:
        f.write(svg)