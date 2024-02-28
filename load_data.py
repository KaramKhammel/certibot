from datasets import load_dataset, Dataset


def load_data(path):
    raw_dataset = load_dataset('json', data_files=path, split='train')
    data = raw_dataset.to_pandas()

    data = data.reset_index()
    data['context'] = data['topic'] + ': ' + data['article']
    data['index'] = data['index'].astype((str))

    dataset = Dataset.from_pandas(data)

    return dataset
