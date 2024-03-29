import pandas as pd
import numpy as np


def read_df(ds_type, root_dir = 'dataset'):

    if ds_type not in ['train', 'test']:
        print("invalid dataset")
        return
    
    ds_type = f'{root_dir}/{ds_type}_dataset.txt'
    
    df = pd.read_csv(ds_type, delimiter='\t')
    
    return df


def train_test_df(root_dir='dataset', names=['train', 'test']):
    return [read_df(name, root_dir) for name in names]


def train_test_mtx(return_dfs = True, return_id_maps = True):
    df, dft = train_test_df()
    df = df.dropna()
    dft = dft.dropna()
    u2i, i2i, R = interaction_matrix(df, True)
    Rt = interaction_matrix(dft, False, u2i, i2i)
    
    result = R, Rt,

    if return_dfs:
        result += df, dft,

    if return_id_maps:
        result += u2i, i2i

    return result


def interaction_matrix(df, return_id_maps=False, u2i=None, i2i=None):

    if not u2i:
        users = sorted(set(df['user_id']))
        user2id = { x:i for i,x in enumerate(users) }
    else:
        user2id = u2i

    if not i2i:
        items = sorted(set(df['item_id']))
        item2id = { x:i for i,x in enumerate(items) }
    else:
        item2id = i2i

    R = np.zeros((len(user2id), len(item2id)))
    
    grouping = df.groupby(by=['user_id', 'item_id'])

    for user, item in grouping.groups.keys():
        user_id = user2id[user]
        item_id = item2id[item]
        R[user_id, item_id] = 1

    if return_id_maps:
        return user2id, item2id, R

    return R


def read_interaction_matrix(dataset = 'train', return_id_maps=False):
    
    df = read_df(dataset)
    
    return interaction_matrix(df, return_id_maps)
