def total_ordering(realizer) -> dict:
    return {
        node: (0, float(i))
        for i, node in enumerate(realizer[0])
    }