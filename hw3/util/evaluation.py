import pandas as pd
import numpy as np
from .data_reader import read_df, interaction_matrix


# init
df_train, df_test = None, None

gold_train = None
gold, K, u_len = None, None, None

Rtr, Rte = None, None


def init():
    global df_train, df_test, gold_train, gold, K, u_len, Rtr, Rte
    
    if df_test is None or df_train is None: 
        print("implict loading of dataframes")

        from util.data_reader import train_test_mtx as train_test
        Rtr, Rte, df_train, df_test, u2i, i2i = train_test()

        _, gold, K, u_len = user_gold(df_test, Rte, u2i, i2i)
        _, gold_train, _, _ = user_gold(df_train, Rtr, u2i, i2i)
        
        print("Max items of a user in test dataset:", K)


def user_gold(df, R, u2i, i2i, max_k=np.inf):
# def user_gold(df, max_k=100):
    '''
    returns
        R: interaction matrix
        gold: dictionary of each user's favorite item indices
        K: max item list size (<= max_k)
        u_len: item list of each user
    '''

    # user2index, item2index, inteRaction matrix
    # u2i, i2i, R = interaction_matrix(df, True)

    u_len = (R > 0).sum(axis=1)
    K = min(u_len.max(), max_k)

    fav = dict([(user, group.tolist()[::-1]) for user, group in df.groupby('user_id')['item_id']])
    gold = { u2i[user]:[i2i[i] for i in items] for user,items in fav.items() }
    gold = { k:np.array(v) for k,v in gold.items() }

    return R, gold, K, u_len



def topk(R, k):
    '''
    returns index of top k items, sorted
    '''

    A = R.argpartition(kth=-k, axis=1)[:, -k:]
    S = np.take_along_axis(R, A, axis=1)
    return np.take_along_axis(A, S.argsort(), axis=1)[:,::-1]


def eval_sorting_for_user(user, fn, guess, gold, u_len):
    '''
    utility function for recall and ndcg for each user
    '''
    
    # n = u_len[user]
    # retrieved = guess[user, :n]
    # gold_n = gold[user][:n]

    retrieved = guess[user, :20]
    gold_n = gold[user]

    # if user == 0:
    #     print(retrieved)
    #     print("------")
    #     print(gold_n)
    
    return fn(gold_n, retrieved)


# utility function

def eval_sorting(gold, guess, u_len, fn):
    return np.array([eval_sorting_for_user(u, fn, guess, gold, u_len) for u in range(len(gold))])
    
# recall

def recall_fn(gold, sorting):
    best = sorting[:20]
    return len(set(best).intersection(gold)) / len(gold)


def recall(gold, guess, u_len):
    return eval_sorting(gold, guess, u_len, recall_fn)


# accuracy

# def acc_fn(gold, guess):
#     best = guess[:20]
#     return len(set(best).intersection(gold)) / len(best)


# def accuracy(gold, guess, u_len):
#     return eval_sorting(gold, guess, u_len, acc_fn)


# dcg

def dcg(gold, guess, return_values=False, p=20):
    p = min(p, len(gold), len(guess))
    guess = guess[:p]

    # relevance
    rel = [int(item in gold) for item in guess][:p]
    rel = np.array(rel) / np.hstack((1, np.log(2 + np.arange(len(rel)-1),)/np.log(2)))
    
    score = rel.sum()
    return (score, rel.cumsum()) if return_values else score


def ndcg_fn(gold, guess, return_values=False, p=20):
    p = min(p, len(gold), len(guess))

    if return_values:
        _, ideal = dcg(gold, gold, return_values=True)
        _, rel = dcg(gold, guess, return_values=True)
        ndcg = (rel/ideal)
        
        return ndcg[0], ndcg

    ideal = dcg(gold, gold)
    rel = dcg(gold, guess)
    
    return rel/ideal


def ndcg(gold, guess, u_len):
    return eval_sorting(gold, guess, u_len, ndcg_fn)


# rank correlation

def rank_correlation_fn(gold, scores):
    '''
    scores: user'th row of scoring matrix S (approximation for R)
    '''

    n = len(gold)
    a = np.arange(n)
    b = (scores[gold]*-1).argsort(kind='mergesort')

    if n == 1:
        return 1 if a[0] == b[0] else 0

    # spearman
    return 1 - (6*(b - a)**2 / (n * (n**2-1))).sum()


def rank_correlation(gold, S):
    n = len(gold)

    scores = [rank_correlation_fn(gold[u], S[u]) for u in range(n)]
    
    return np.array(scores)


# remove items seen in train

def remove_train_items(S):
    global gold_train
    S = S.copy()
   
    for u, items in gold_train.items():
       S[u, items] = -np.inf 

    return S


# evaluation handle

def evaluate(S, df_test=df_test, df_train=df_train, report_average=True, format=True):
    '''
    S: Score matrix (estimated R)
    '''

    global gold, K, u_len, gold_train

    S = remove_train_items(S)
    guess = topk(S, K)
    # print(guess.shape)
    
    result = {
        # 'accuracy': 100 * accuracy(gold, guess, u_len),
        'recall': 100 * recall(gold, guess, u_len),
        'ndcg': ndcg(gold, guess, u_len),
        'rank correlation': rank_correlation(gold, S)
    }

    
    if report_average:
        result = {k:v.mean() for k,v in result.items()}

        if format:
            result = {k: f'{v:.3f}' if v < 10 else f'{int(v)}' for k,v in result.items()}
    
    return result