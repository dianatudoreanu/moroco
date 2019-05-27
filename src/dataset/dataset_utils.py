from src.utils.paths_utils import get_absolute_path
from src.utils.string_utils import clean_string
import os

class DatasetUtils:

    @staticmethod
    def get_sk_indices(sk_path):
        sk_path = get_absolute_path(sk_path)

        sk_indices = open(sk_path).read().split('\n')
        if len(sk_indices[-1]) == 0:
            sk_indices = sk_indices[:-1]

        sk_indices = list(map(int, sk_indices))
        
        return sk_indices

    @staticmethod
    def split_val_test(path, output_val_path, output_test_path, sk_path,
                       split_label_ratio=0.5):
        texts, labels = DatasetUtils.read_raw_dataset(path)
        texts = list(map(clean_string, texts))
        
        sk_indices = DatasetUtils.get_sk_indices(sk_path)
        
        label_dict = {}
        for index, label in enumerate(labels):
            if label not in label_dict:
                label_dict[label] = []

            label_dict[label].append((texts[index], sk_indices[index]))

        val_dataset = []
        test_dataset = []
        for label, values in label_dict.items():
            texts, sk_indices = list(zip(*values))

            split_index = int(round(len(texts) * split_label_ratio))
            
            val_dataset += list(zip(texts[:split_index], [label] * split_index, sk_indices[:split_index]))
            test_dataset += list(zip(texts[split_index:], [label] * (len(texts) - split_index), sk_indices[split_index:]))

        DatasetUtils.write_dataset(val_dataset, output_val_path)
        DatasetUtils.write_dataset(test_dataset, output_test_path)

    
    @staticmethod
    def convert_to_dataset(input_path, output_path, sk_path):
        texts, labels = DatasetUtils.read_raw_dataset(input_path)
        texts = list(map(clean_string, texts))

        sk_indices = DatasetUtils.get_sk_indices(sk_path)

        result_dataset = list(zip(texts, labels, sk_indices))

        DatasetUtils.write_dataset(result_dataset, output_path)

    @staticmethod
    def read_raw_dataset(path):
        path = get_absolute_path(path)

        texts = open(path, encoding="utf8").read().split('\n')
        if len(texts[-1]) == 0:
            texts = texts[:-1]

        if '\t' in texts[0]:        
            texts, labels = list(zip(*list(map(lambda x: tuple(x.split('\t')), texts))))
        else:
            labels = ['UNK'] * len(texts)
        
        return list(map(clean_string, texts)), labels

    @staticmethod
    def read_dataset(path, return_tuples = True):
        path = get_absolute_path(path)

        texts = open(path, encoding="utf8").read().split('\n')
        if len(texts[-1]) == 0:
            texts = texts[:-1]

        if return_tuples is True:
            def split_entity(entity):
                text, label, sk_index = entity.split('\t')
                sk_index = int(sk_index)

                return [text, label, sk_index]

            return list(map(split_entity, texts))
        else:
            texts, labels, sk_indices = list(map(list, list(zip(*list(map(lambda x: tuple(x.split('\t')), texts))))))
            sk_indices = list(map(int, sk_indices))

            return texts, labels, sk_indices
    
    @staticmethod
    def write_dataset(data, output_path):
        output_path = get_absolute_path(output_path)

        with open(output_path, 'w', encoding="utf8") as f:
            for text, label, sk_index in data:
                f.write(f'{text}\t{label}\t{sk_index}\n')


def join_datasets():
    tr_data = DatasetUtils.read_dataset('@data/subtask1/train_dataset.txt')
    d_data = DatasetUtils.read_dataset('@data/subtask1/dev_dataset.txt')
    
    test_path = "@data/MOROCO/MOROCO/TESTSET-MRC-subtasks-1+2+3-VARDIAL2019/TESTSET-MRC-subtasks-1+2+3-VARDIAL2019/test.txt"
    t_texts, t_labels = DatasetUtils.read_raw_dataset(test_path)
    t_texts = list(map(clean_string, t_texts))
    t_sk_indices = [-1] * len(t_texts)

    t_data = list(zip(t_texts, t_labels, t_sk_indices))

    joined_dataset = tr_data + d_data + t_data
    output_path = "@data/all_dataset.txt"

    DatasetUtils.write_dataset(joined_dataset, output_path)

if __name__ == '__main__':
    task = "subtask3"
    DatasetUtils.split_val_test(f'@data/{task}/dev.txt', 
                                   f'@data/{task}/val_dataset.txt',
                                   f'@data/{task}/test_dataset.txt',
                                   sk_path=f'@data/{task}/dev_ids.txt')
    
    DatasetUtils.convert_to_dataset(f'@data/{task}/train.txt',
                                       f'@data/{task}/train_dataset.txt',
                                       sk_path=f'@data/{task}/train_ids.txt')

    DatasetUtils.convert_to_dataset(f'@data/{task}/dev.txt',
                                       f'@data/{task}/dev_dataset.txt',
                                       sk_path=f'@data/{task}/dev_ids.txt')


    #just for testing
    # DatasetUtils.split_val_test('@data/subtask1/train.txt', 
    #                                '@data/subtask1/train_val_dataset.txt',
    #                                '@data/subtask1/train_test_dataset.txt',
    #                                sk_path='@data/subtask1/train_ids.txt')

    # texts, labels, sk_indices = DatasetUtils.read_dataset('@data/subtask1/train_dataset.txt')

    #join datasets to @data/all_dataset.txt
    join_datasets()