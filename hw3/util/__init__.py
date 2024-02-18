from .data_reader import train_test_df, interaction_matrix, train_test_mtx
from .evaluation import evaluate, topk
from .plot import heatmaps


df_test = data_reader.read_df('test')
evaluation.init()

def easy_eval(M):
    return evaluate(M, df_test)