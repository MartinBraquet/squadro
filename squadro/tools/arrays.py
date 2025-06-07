def box(x):
    if isinstance(x, list):
        return x
    return [x]


def unbox(x):
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x
