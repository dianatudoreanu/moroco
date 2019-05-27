from src.utils.paths_utils import get_absolute_path
from src.utils.compute_utils import compute_string_kernel
from src.dataset.dataset_utils import DatasetUtils
from src.utils.string_utils import ALPHABET, BLANK_CHAR, NE_CHAR
from multiprocessing import Process, Queue, RawArray
from random import shuffle, randint
import numpy as np
import os, json


def split_entity(entity, string_size):
    text, label, kernel_id = entity
    text_length = len(text)

    if text_length > string_size:
        kernel_id = None

    if text_length > string_size:
        text_length -= text_length % string_size

    result = []
    for i in range(0, text_length, string_size):
        result.append((text[i:i+string_size], label, kernel_id))
    
    return result

def split_text(text, string_size, label=None, kernel_id=None):
    text_length = len(text)

    if text_length > string_size:
        kernel_id = None

    if text_length > string_size:
        text_length -= text_length % string_size

    result = []
    for i in range(0, text_length, string_size):
        result.append((text[i:i+string_size], label, kernel_id))
    
    return result
    
def decode_one_hot(one_hot):
    result = ''

    for index in range(one_hot.shape[0]):
        search = np.where(one_hot[index] == 1)[0]
        if search.shape[0] == 1:
            result += ALPHABET[search[0]]

    return result

def decode_ascii_encoding(ascii_encoding):
    result = ''

    for index in range(ascii_encoding.shape[0]):
        if int(ascii_encoding[index]) > 0:
            result += ALPHABET[int(ascii_encoding[index]) - 1]

    return result

def get_one_hot_string(text, string_size):
    one_hot = np.zeros((string_size, len(ALPHABET)), dtype=np.float32)

    for index, token in enumerate(text):
        one_hot[index, ALPHABET.index(token)] = 1

    return one_hot

def get_ascii_encoding(text, string_size):
    ascii_encoding = np.full((string_size, 1), fill_value=0, dtype=np.float32)

    for index, token in enumerate(text):
        ascii_encoding[index] = ALPHABET.index(token) + 1
    # ascii_encoding /= len(ALPHABET)

    return ascii_encoding   

def get_string_kernel(sk, entity1, entity2):
    if entity1[2] is not None and entity2[2] is not None:
        i1 = entity1[2]
        i2 = entity2[2]

        return float(sk[i1, i2]) / max(1, np.sqrt(float(sk[i1, i1]) * float(sk[i2, i2])))
    else:
        t1 = entity1[0]
        t2 = entity2[0]
        result = float(compute_string_kernel(t1, t2))
        result /= max(1, np.sqrt(float(compute_string_kernel(t1, t1)) * float(compute_string_kernel(t2, t2))))

        return result

def batch_process(dataset_path, string_kernel_path, queue, 
                  string_size, batch_size, epochs, shuffle_data, 
                  encoding_type, label_type):
    if 'SIAMESE' in label_type:
        assert(batch_size % 2 == 0)


    entities = DatasetUtils.read_dataset(dataset_path)
    string_kernel = np.load(get_absolute_path(string_kernel_path))

    sk_indices = [entity[2] for entity in entities]
    sk_indices = list(set(sk_indices))
    sk_indices.sort()
    sk_dict = {x: i for i, x in enumerate(sk_indices)}

    for index in range(len(entities)):
        entities[index][2] = sk_dict[entities[index][2]]

    data = []
    for entity in entities:
        data += split_entity(entity, string_size)

    if label_type == 'BINARY':
        labels_list = list(set(map(lambda x: x[1], data)))
        labels_list.sort()
        labels_dict = {}
        for i, k in enumerate(labels_list):
            label_np = np.zeros((len(labels_list),), dtype=np.float32)
            label_np[i] = 1
            labels_dict[k] = label_np

        assert(batch_size % len(labels_list) == 0)

        it_index = 0
        epoch_index = 0
        while epochs is None or epoch_index < epochs:
            if shuffle_data is True:
                shuffle(data)

            data_per_label = {}
            min_length = 1000000
            for key in labels_dict:
                data_per_label[key] = [entity for entity in data if entity[1] == key]
                min_length = min(min_length, len(data_per_label[key]))

            batch_size_label = batch_size // len(data_per_label)
            min_length = min_length - min_length % batch_size_label

            for batch_index in range(0, min_length, batch_size_label):
                batch_data = []
                batch_labels = []

                for key in labels_dict:
                    if encoding_type == 'ONE_HOT':
                        batch_data += list(map(lambda x: get_one_hot_string(x[0], string_size),
                                            data_per_label[key][batch_index:batch_index + batch_size_label]))
                    else:
                        batch_data += list(map(lambda x: get_ascii_encoding(x[0], string_size),
                                            data_per_label[key][batch_index:batch_index + batch_size_label]))
                    batch_labels += list(map(lambda x: labels_dict[x[1]],
                                             data_per_label[key][batch_index:batch_index + batch_size_label]))

                batch_data = np.expand_dims(np.array(batch_data, dtype=np.float32), -1)
                batch_labels = np.array(batch_labels, dtype=np.float32)

                it_index += 1 
                if batch_index + 2 * batch_size_label > min_length:
                    queue.put((batch_data, batch_labels, epoch_index))
                else:
                    queue.put((batch_data, batch_labels))

            epoch_index += 1
    elif "TRIPLET_LOSS" in label_type:
        labels_list = list(set(map(lambda x: x[1], data)))
        labels_list.sort()
        labels_dict = {}
        for i, k in enumerate(labels_list):
            label_np = np.zeros((len(labels_list),), dtype=np.float32)
            label_np[i] = 1
            labels_dict[k] = label_np

        assert(batch_size % (3 * len(labels_list)) == 0)

        it_index = 0
        epoch_index = 0
        while epochs is None or epoch_index < epochs:
            if shuffle_data is True:
                shuffle(data)

            data_per_label = {}
            min_length = 1000000
            for key in labels_dict:
                data_per_label[key] = [entity for entity in data if entity[1] == key]
                min_length = min(min_length, len(data_per_label[key]))
            
            batch_size_label = batch_size // (3 * len(data_per_label))
            min_length = min_length - min_length % batch_size_label
            for batch_index in range(0, min_length, batch_size_label):
                batch_data = []
                batch_labels = []
                for key in labels_dict:
                    for index in range(batch_size_label):
                        triplet = []
                        triplet.append(data_per_label[key][batch_index + index])

                        pos_index = randint(0, len(data_per_label[key]) - 1)
                        while pos_index == batch_index + index:
                            pos_index = randint(0, len(data_per_label[key]) - 1)
                        triplet.append(data_per_label[key][pos_index])

                        neg_labels = list(labels_dict.keys())
                        neg_labels.remove(key)
                        neg_label = neg_labels[randint(0, len(neg_labels) - 1)]
                        
                        neg_index = randint(0, len(data_per_label[neg_label]) - 1)
                        triplet.append(data_per_label[neg_label][neg_index])

                        batch_labels.append(key)
                        batch_labels.append(key)
                        batch_labels.append(neg_label)

                        if encoding_type == 'ONE_HOT':
                            batch_data += list(map(lambda x: get_one_hot_string(x[0], string_size),
                                            triplet))
                        else:
                            batch_data += list(map(lambda x: get_ascii_encoding(x[0], string_size),
                                            triplet))

                batch_data = np.expand_dims(np.array(batch_data, dtype=np.float32), -1)

                if batch_index + 2 * batch_size_label > min_length:
                    queue.put((batch_data, batch_labels, epoch_index))
                else:
                    queue.put((batch_data, batch_labels))            
    else:
        epoch_index = 0
        while epochs is None or epoch_index < epochs:
            if shuffle_data is True:
                shuffle(data)

            total_size = len(data) - len(data) % batch_size
            for batch_index in range(0, total_size, batch_size):
                #process batch
                batch_data = []
                batch_labels = []

                for index in range(batch_index, batch_index + batch_size, 2):
                    entity1 = data[index]
                    entity2 = data[index + 1]

                    if encoding_type == 'ONE_HOT':
                        batch_data.append(get_one_hot_string(entity1[0], string_size))
                        batch_data.append(get_one_hot_string(entity2[0], string_size))
                    else:
                        batch_data.append(get_ascii_encoding(entity1[0], string_size))
                        batch_data.append(get_ascii_encoding(entity2[0], string_size))

                    if label_type == 'SIAMESE_STRING_KERNEL':
                        batch_labels.append(get_string_kernel(string_kernel, entity1, entity2))
                    elif label_type == 'SIAMESE_BINARY':
                        batch_labels.append(0 if entity1[1] == entity2[1] else 1)

                batch_data = np.expand_dims(np.array(batch_data, dtype=np.float32), -1)
                batch_labels = np.array(batch_labels, dtype=np.float32)
                
                if batch_index + 2 * batch_size > len(data):
                    queue.put((batch_data, batch_labels, epoch_index))
                else:
                    queue.put((batch_data, batch_labels))            

            epoch_index += 1


class DatasetGenerator:

    def __init__(self, config_path):
        self.params = json.load(open(get_absolute_path(config_path)))
        self.string_size = self.params["string_size"]

        self.train_queue = None
        self.val_queue = None
        self.test_queue = None

        self.__prepare_string_kernels()

    def __cache_sub_string_kernel(self, string_kernel, dataset_path, cache_path):
        sk_indices = DatasetUtils.read_dataset(dataset_path,
                                                    return_tuples=False)[2]
        sk_indices = list(set(sk_indices))
        sk_indices.sort()
        sk_indices = np.array(sk_indices)

        np.save(get_absolute_path(cache_path), string_kernel[sk_indices[:, None], sk_indices])

    def __prepare_string_kernels(self):
        self.normalize_string_kernel = self.params['normalize_string_kernel']
        string_kernel = np.load(get_absolute_path(self.params["string_kernel_path"]))

        self.__cache_sub_string_kernel(string_kernel,
                                        self.params["train_dataset_path"],
                                        self.params["string_kernel_cache_train_path"])
        
        self.__cache_sub_string_kernel(string_kernel,
                                        self.params["val_dataset_path"],
                                        self.params["string_kernel_cache_val_path"])

        self.__cache_sub_string_kernel(string_kernel,
                                        self.params["test_dataset_path"],
                                        self.params["string_kernel_cache_test_path"])


    def get_train_generator(self):
        self.train_queue = Queue(maxsize=self.params['queue_limit'])
        self.train_process = Process(target=batch_process,
                                     args=(self.params['train_dataset_path'],
                                           self.params['string_kernel_cache_train_path'],
                                           self.train_queue,
                                           self.string_size,
                                           self.params['train_batch_size'],
                                           self.params['train_no_epochs'],
                                           self.params['train_shuffle'],
                                           self.params['encoding_type'],
                                           self.params['label_type']))
        
        return self.train_process, self.train_queue

    
    def get_validation_generator(self):
        self.val_queue = Queue(maxsize=self.params['queue_limit'])
        self.val_process = Process(target=batch_process,
                                   args=(self.params['val_dataset_path'],
                                         self.params['string_kernel_cache_val_path'],
                                         self.val_queue,
                                         self.string_size,
                                         self.params['val_batch_size'],
                                         self.params['val_no_epochs'],
                                         self.params['val_shuffle'],
                                         self.params['encoding_type'],
                                         self.params['label_type']))
        
        return self.val_process, self.val_queue

    def get_test_generator(self):
        self.test_queue = Queue(maxsize=self.params['queue_limit'])
        self.test_process = Process(target=batch_process,
                                   args=(self.params['test_dataset_path'],
                                         self.params['string_kernel_cache_test_path'],
                                         self.test_queue,
                                         self.string_size,
                                         self.params['test_batch_size'],
                                         self.params['test_no_epochs'],
                                         self.params['test_shuffle'],
                                         self.params['encoding_type'],
                                         self.params['label_type']))
        
        return self.test_process, self.test_queue



if __name__ == '__main__':
    config_path = '@src/config/dataset_generator_config.json'

    dg = DatasetGenerator(config_path)
    
    train_proc, train_queue = dg.get_train_generator()
    train_proc.start()

    val_proc, val_queue = dg.get_validation_generator()
    val_proc.start()

    test_proc, test_queue = dg.get_test_generator()
    test_proc.start()

    import pdb; pdb.set_trace()



