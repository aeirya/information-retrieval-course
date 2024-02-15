import pandas as pd
import numpy as np

def read_df(ds_type):

    if ds_type not in ['train', 'test']:
        print("invalid dataset")
        return
    
    ds_type = f'dataset/{ds_type}_dataset.txt'
    df = pd.read_csv(ds_type, delimiter='\t')
    
    return df


def interaction_matrix(df, return_id_maps=True):

    users = sorted(set(df['user_id']))
    items = sorted(set(df['item_id']))

    grouping = df.groupby(by=['user_id', 'item_id'])
    user_to_id = { x:i for i,x in enumerate(users) }
    item_to_id = { x:i for i,x in enumerate(items) }

    R = np.zeros((len(users), len(items)))

    for user, item in grouping.groups.keys():
        user_id = user_to_id[user]
        item_id = item_to_id[item]
        R[user_id, item_id] = 1

    if return_id_maps:
        return user_to_id, item_to_id, R

    return R


def read_interaction_matrix(dataset = 'train', return_id_maps=False):
    
    df = read_df(dataset)
    
    return interaction_matrix(df, return_id_maps)
