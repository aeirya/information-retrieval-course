import re

patterns = {index: re.compile(rf'<e{index}>.*</e{index}>') for index in range(1, 3)}


def remove_entities(text):
    tag_pattern = r'</?e\d>'
    text = re.sub(tag_pattern, '', text)
    return text


def get_entity(text, index):
    ei = patterns[index].findall(text)[0]
    return ei[4:-5].strip()


def extract_entities(text):
    return {
        'e1': get_entity(text, 1),
        'e2': get_entity(text, 2),
    }


__index_finder_patten__ = re.compile(r'^\d*')

def read_sentence(raw_text):

    index = __index_finder_patten__.findall(raw_text)[0]
    
    # remove index, and "s in the beginning and end
    raw = raw_text[len(index):].strip()[1:-1]
    
    index = int(index)
    
    entities = extract_entities(raw)
    clean_text = remove_entities(raw).strip()

    return {
        'entities': entities,
        'index': index,
        'text': clean_text,
        'raw': raw,
    }


def read_relation(text):
    
    if text == 'Other':
        return {
            'relation': 'Other'
        }
    
    args = re.search('\(.*\)', text)
    is_reversed = re.findall('e\d', args.group())[0] == 'e2'
    relation = text[:args.start()]
    
    return {
        'relation': relation,
        'is_reversed': is_reversed
    }


def read_sample(lines):
    '''
    input: three lines of text
    '''

    # dev todo note: also parse comments and translations
    raw_text, raw_relation, _ = lines

    # remove half-space for better output readability
    raw_text = raw_text.replace('\u200c', ' ')

    return {
        **read_relation(raw_relation),
        **read_sentence(raw_text),
    }


def flatten(data):
    for sample in data:
        entities = sample.pop('entities')
        e1, e2 = entities.values()
        sample['e1'] = e1
        sample['e2'] = e2

    return data


def pd_format(data):
    # WARNING: split entities is assumed

    import pandas as pd
    df = pd.DataFrame(data)

    # rearranging the columns
    return df[['index', 'text', 'relation', 'e1', 'e2', 'is_reversed', 'raw']]


def read_dataset(path, split_entities=True, return_df=True):
    '''
    split_entities:
        flattens depth of the tree 
        and stores entities in different columns

    return_df:
        returns pandas DataFrame as output
    '''

    data = []

    with open(path, 'r') as file:
        while True:
            # each sample takes three lines
            lines = [file.readline().strip() for _ in range(3)]
            
            if len(lines[-1]) == 0:
                break

            data.append(read_sample(lines))
            
            # read empty line
            line = file.readline()

            if len(line) == 0:
                break

    if split_entities or return_df:
        data = flatten(data)

    if return_df:
        return pd_format(data)

    return data


