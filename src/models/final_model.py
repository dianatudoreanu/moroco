from src.utils.paths_utils import get_absolute_path
from src.dataset.dataset_utils import DatasetUtils
from src.dataset.dataset_generator import get_ascii_encoding
from src.utils.compute_utils import compute_string_kernel
from sklearn.svm import SVC
import tensorflow as tf
import numpy as np
import json, os

DIALECT_DICT = {
    "MD": 0,
    "RO": 1
}
DIALECT_LIST = ["MD", "RO"]

TOPICS_DICT = {
    "CUL": 0,
    "FIN": 1,
    "POL": 2,
    "SCI": 3,
    "SPO": 4,
    "TEC": 5
}
TOPICS_LIST = ["CUL", "FIN", "POL", "SCI", "TEC"]

LIMIT = 5000
LIMIT_LENGTH = 5000

def model_sk(config):
    sk_matrix = np.load(get_absolute_path(config['sk_path'])).astype(np.float32)
    dp = np.reshape(np.diag(sk_matrix), (sk_matrix.shape[0], 1))
    dp_t = dp.T
    num = np.matmul(dp, dp_t) 
    num[num == 0] = 1
    sk_matrix = sk_matrix/np.sqrt(num)

    train_data, train_labels, train_sk_indices = DatasetUtils.read_dataset(config['train_dataset_path'],
                                                                        return_tuples=False)
    val_data, val_labels, val_sk_indices = DatasetUtils.read_dataset(config['dev_dataset_path'],
                                                                        return_tuples=False)
    test_data, _ = DatasetUtils.read_dataset(config['test_dataset_path'], return_tuples=False)

    if config['data_type'] == 'binary':
        train_labels = np.array([[1, -1] if x == 'MD' else [-1, 1] for x in train_labels[:lens]])
        val_labels = np.array([[1, -1] if x == 'MD' else [-1, 1] for x in val_labels])
    else:
        dialect_np_dict = {}
        for key, value in DIALECT_DICT:
            np_array = np.zeros((len(DIALECT_DICT),), dtype=np.float32)
            np_array[value] = 1.0
            dialect_np_dict[key] = np_array
        train_labels = np.array([dialect_np_dict[x[1]] for x in train_labels[:LENGTH]])
        val_labels = np.array([dialect_np_dict[x[1]] for x in train_labels[:LENGTH]])

    train_krr_data = np.array(list(map( lambda x : sk_matrix[x][train_sk_indices], train_sk_indices))).astype(np.float32)
    val_krr_data = np.array(list(map( lambda x : sk_matrix[x][train_sk_indices], val_sk_indices))).astype(np.float32)

    test_krr_data = [] 
    for test_text in test_data:
        test_krr_data.append(list(map(lambda text: compute_string_kernel(text, test_text), train_data)))
    test_krr_data = np.array(test_krr_data)

    lam = 0.00001
    (w,a,b,c) =  np.linalg.lstsq(train_krr_data + np.eye(n) * n * lam, train_labels)
    
    train_ans = np.argmax(train_labels, 1)
    train_preds = np.matmul(train_krr_data, w)
    train_preds_ans = np.argmax(train_preds, 1)
    train_acc = np.sum(train_preds_ans == train_ans) / train_ans.shape[0]

    val_ans = np.argmax(val_labels, 1)
    val_preds = np.matmul(val_krr_data, w)
    val_preds_ans = np.argmax(val_preds, 1)
    val_acc = np.sum(val_preds_ans == val_ans) / val_ans.shape[0]
    
    test_preds = np.matmul(val_krr_data, w)

    print(f'ModelSK: Train acc: {train_acc}')
    print(f'ModelSK: Val acc: {val_acc}')

    return train_preds, train_labels, val_preds, val_labels, test_preds

def normalize_text(text):
    #limit to first LIMIT_LENGTH
    return get_ascii_encoding(text[:LIMIT_LENGTH], LIMIT_LENGTH)

def use_model(sess, data, input_op, output_op, expand_dims = True):
    batch_size = 100
    result = []
    if (expand_dims):
        data = np.expand_dims(data, -1)
    for index in range(0, data.shape[0], batch_size):
        result.append(sess.run(output_op, feed_dict = {input_op: data[index:index+batch_size]}))
    result = np.concatenate(result)
    return result

def get_labels(labels):
    labels_values = list(set(labels))
    labels_values.sort()
    print(labels_values)
    if len(labels_values) == 2:
        labels_values_dict = {}
        labels_values_dict[labels_values[0]] = -1
        labels_values_dict[labels_values[1]] = 1
    else:
        labels_values_dict = {}
        labels_values_dict[labels_values[0]] = 0
        labels_values_dict[labels_values[1]] = 1
        labels_values_dict[labels_values[2]] = 2
        labels_values_dict[labels_values[3]] = 3
        labels_values_dict[labels_values[4]] = 4
        labels_values_dict[labels_values[5]] = 5
    labels = np.array(list(map(lambda x: labels_values_dict[x], labels)))
    return labels

def model_emb(config):
    train_data, train_labels, _ = DatasetUtils.read_dataset(config['train_dataset_path'],
                                                                        return_tuples=False)
    val_data, val_labels, _ = DatasetUtils.read_dataset(config['dev_dataset_path'],
                                                                        return_tuples=False)
    test_data, _ = DatasetUtils.read_raw_dataset(config['test_dataset_path'])

    train_data, train_labels = train_data[:LIMIT], train_labels[:LIMIT]
    val_data, val_labels = val_data[:LIMIT], val_labels[:LIMIT]
    
    train_data = np.array(list(map(normalize_text, train_data)))
    val_data = np.array(list(map(normalize_text, val_data)))
    test_data = np.array(list(map(normalize_text, test_data)))

    train_labels = np.array(get_labels(train_labels))
    val_labels = np.array(get_labels(val_labels))

    tf.reset_default_graph()
    model_path = get_absolute_path(config['triplet_model_path'])
    saver = tf.train.import_meta_graph(model_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        input_op = tf.get_default_graph().get_tensor_by_name("input:0")
        emb_op = tf.get_default_graph().get_tensor_by_name("embedding_dense_layer/LeakyRelu:0")

        train_emb = use_model(sess, train_data, input_op, emb_op)
        val_emb = use_model(sess, val_data, input_op, emb_op)
        test_emb = use_model(sess, test_data, input_op, emb_op)

    if config['data_type'] == 'binary':
        model = SVC(kernel=config['emb_kernel'], C=config['emb_C'])
    else:
        model = SVC(kernel=config['emb_kernel'], C=config['emb_C'], decision_function_shape = 'ovo')
    print (train_emb, train_labels)
    model.fit(train_emb, train_labels)

    train_preds_ans = model.predict(train_emb)
    train_preds = model.decision_function(train_emb)
    train_acc = np.sum(train_preds_ans == train_labels) / train_labels.shape[0]


    val_preds_ans = model.predict(val_emb)
    val_preds = model.decision_function(val_emb)
    val_acc = np.sum(val_preds_ans == val_labels) / val_labels.shape[0]

    test_preds = model.decision_function(test_emb)

    print(f"ModelEmb: Train accuracy: {train_acc} Val Accuracy: {val_acc}")
    
    return train_preds, train_labels, val_preds, val_labels, test_preds
    # return np.expand_dims(train_preds, -1), train_labels, np.expand_dims(val_preds, -1), val_labels, np.expand_dims(test_preds, -1)
    

def model_classifier(config):
    tf.reset_default_graph()

    model_path = get_absolute_path(config['classifier_model_path'])

    train_data, train_labels, _ = DatasetUtils.read_dataset(config['train_dataset_path'],
                                                                        return_tuples=False)
    val_data, val_labels, _ = DatasetUtils.read_dataset(config['dev_dataset_path'],
                                                                        return_tuples=False)
    test_data, _ = DatasetUtils.read_raw_dataset(config['test_dataset_path'])

    train_data, train_labels = train_data[:LIMIT], train_labels[:LIMIT]
    val_data, val_labels = val_data[:LIMIT], val_labels[:LIMIT]
    
    train_data = np.array(list(map(normalize_text, train_data)))
    val_data = np.array(list(map(normalize_text, val_data)))
    test_data = np.array(list(map(normalize_text, test_data)))

    train_labels = np.array(get_labels(train_labels))
    val_labels = np.array(get_labels(val_labels))

    saver = tf.train.import_meta_graph(model_path + '.meta')
    
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        input_op = tf.get_default_graph().get_tensor_by_name("input:0")
        output_op = tf.get_default_graph().get_tensor_by_name("Softmax:0")

        train_preds = use_model(sess, train_data, input_op, output_op)
        val_preds = use_model(sess, val_data, input_op, output_op)
        test_preds = use_model(sess, test_data, input_op, output_op)

    train_ans = np.argmax(train_preds, 1)
    train_acc = np.sum(train_ans == train_labels) / train_ans.shape[0]

    val_ans = np.argmax(val_preds, 1)
    val_acc = np.sum(val_ans == val_labels) / val_ans.shape[0]

    print(f"ModelClassifier: Train accuracy: {train_acc} Val accuracy: {val_acc}")

    return train_preds, train_labels, val_preds, val_labels, test_preds

def rev_labels(labels, config):
    if config['data_type'] == 'binary':
        return list(map(lambda x: (x+1)/2 + 1, labels))
    else:
        return list(map(lambda x: (x+1), labels))
def model_ensemble(config):

    train_data = []
    val_data = []
    test_data = []

    # train_preds, train_labels, val_preds, val_labels, test_preds = model_sk(config)
    # train_data.append(train_preds)
    # val_data.append(val_preds)
    # test_data.append(test_preds)

    train_preds, train_labels, val_preds, val_labels, test_preds = model_classifier(config)
    train_data.append(train_preds)
    val_data.append(val_preds)
    test_data.append(test_preds)

    train_preds, train_labels, val_preds, val_labels, test_preds = model_emb(config)
    train_data.append(train_preds)
    val_data.append(val_preds)
    test_data.append(test_preds)

    print(list(map(lambda x: x.shape, train_data)))
    train_data = np.hstack(train_data)
    val_data = np.hstack(val_data)
    test_data = np.hstack(test_data)

    if config['data_type'] == 'binary':
        model = SVC(kernel=config['ensemble_kernel'], C=config['ensemble_C'])
    else:
        model = SVC(kernel=config['ensemble_kernel'], C=config['ensemble_C'], decision_function_shape = 'ovo')
    model.fit(train_data, train_labels)

    train_preds = model.predict(train_data)
    train_acc = np.sum(train_preds == train_labels) / train_labels.shape[0]

    val_preds = model.predict(val_data)
    val_acc = np.sum(val_preds == val_labels) / val_labels.shape[0]

    test_preds = model.predict(test_data)
    test_preds = rev_labels(test_preds, config)
    print(f'Final ensemble: Train accuracy: {train_acc} Val accuracy: {val_acc}')

    with open(get_absolute_path(config['result_path']), 'w') as f:
        for label in test_preds:
            f.write(f'{label}\n')


if __name__ == '__main__':
    config_path = '@src/config/final_model_config.json'
    config = json.load(open(get_absolute_path(config_path)))
    model_ensemble(config)