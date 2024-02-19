from .data_reader import train_test_df, interaction_matrix, train_test_mtx, read_df
from .evaluation import evaluate, topk
from .plot import heatmaps, heatmap

import numpy as np


# df_test = eval_init()
evaluation.init()

# def easy_eval(M):
#     return evaluate(M, df_test)


def repeat_like(a, M):
    '''
    a(n,) and M(n,m)
    '''

    a = np.repeat(a, M.shape[1]).reshape((a.shape[0], -1))
    return a


def normal(M):
    a = ((M.min(axis=1) == 0) & (M.max(axis=1) == 0))
    
    M = M.copy()

    M -= repeat_like(M.min(axis=1), M)
    M[~a, :] /= repeat_like(M.max(axis=1), M)[~a, :]
    
    M = np.where(repeat_like(a, M), 0, M)

    return M

def normal_exp(M):
    '''
    map values to the [0,1] range
    '''

    M = M.copy()
    M = normal(M)
    M = np.exp(M)

    return normal(M)

