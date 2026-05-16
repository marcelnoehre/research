type_mapping = {
    'hand_drawn': 'hand_drawn', 
    'sup_inf_attribute': 'sup_inf_attribute', 
    'sup_inf_double': 'sup_inf_doubly', 
    'dim_draw': 'dim_draw', 
    'dim_draw_double': 'dim_flux'
}
cxt_mapping = {
    'living_beings_and_water': 'living_beings_and_water', 
    'car': 'drive_concepts', 
    'Forum-Romanum': 'forum_romanum',
    'triangles': 'triangles' , 
    'convex-ordinal': 'convex_ordinal'
}
for type_original in [
    'hand_drawn', 
    'sup_inf_attribute', 
    'sup_inf_double', 
    'dim_draw', 
    'dim_draw_double'
    ]:
    for cxt_original in [i for i in range(1, 127)] + [
        'living_beings_and_water', 
        'car', 
        'Forum-Romanum',
        'triangles', 
        'convex-ordinal'
    ]:
        pass