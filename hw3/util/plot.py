def heatmap(a):
    import matplotlib.pyplot as plt
    plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show()


def heatmaps(iterable, devide_n = 0.5):
    import numpy as np
    from functools import reduce

    n = iterable[0].shape[0]
    D = np.ones((n, 1)) * devide_n
    
    t = reduce(lambda a,b:a+b, [[M, D] for M in iterable], [])
    T = np.concatenate(t, axis=1)
    heatmap(T)