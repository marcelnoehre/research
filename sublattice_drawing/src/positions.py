def total_ordering(realizer):
    return {
        node: (0, i)
        for i, node in enumerate(realizer)
    }