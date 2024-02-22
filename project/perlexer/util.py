def make_dataset(train_df, test_df):

    from datasets.dataset_dict import DatasetDict
    from datasets import Dataset

    ds = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df),
    })

    return ds


def load_in_pandas(dir):

    from perlexer import read_data
    train, test = [read_data(path, return_df=True) for path in [f'{dir}/{name}.txt' for name in ['train', 'test']]]

    return train, test


def load_dataset(dir = 'PERLEX'):

    return make_dataset(*load_in_pandas(dir))


def push_to_hf(load_dir= None):

    ds = load_dataset(load_dir) if load_dir else load_dataset()
    ds.push_to_hub('Aeirya/PERLEX')

