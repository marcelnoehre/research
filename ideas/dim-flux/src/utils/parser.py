from fcapy.context import FormalContext

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

