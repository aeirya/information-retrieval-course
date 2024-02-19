from .data_reader import train_test_df, interaction_matrix, train_test_mtx, read_df
from .evaluation import evaluate, topk
from .plot import heatmaps, heatmap


# df_test = eval_init()
evaluation.init()

# def easy_eval(M):
#     return evaluate(M, df_test)