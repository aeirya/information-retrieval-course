from .data_reader import train_test_df, interaction_matrix, train_test_mtx, read_df
from .evaluation import evaluate, topk, init as eval_init
from .plot import heatmaps, heatmap


df_test = read_df('test')
eval_init()

def easy_eval(M):
    return evaluate(M, df_test)