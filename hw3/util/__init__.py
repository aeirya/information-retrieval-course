from .data_reader import train_test_df, interaction_matrix
from .evaluation import evaluate
from .plot import heatmaps


df_test = data_reader.read_df('test')

def easy_eval(M):
    return evaluate(M, df_test)