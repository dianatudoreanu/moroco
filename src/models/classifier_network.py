from src.models.abstract_neural_network import AbstractNeuralNetwork
from src.models.models_utils import create_layer
from src.utils.paths_utils import get_absolute_path
from src.dataset.dataset_generator import DatasetGenerator, get_one_hot_string, split_text
from src.dataset.dataset_generator import decode_one_hot, get_ascii_encoding, decode_ascii_encoding
from src.dataset.dataset_utils import DatasetUtils
from src.utils.string_utils import clean_string
from sklearn.svm import SVC
from tqdm import tqdm
import tensorflow as tf
import json
import numpy as np
import os

class ClassifierNetwork(AbstractNeuralNetwork):

    def __init__(self, config_path):
        self.config_path = get_absolute_path(config_path)
        self.config = json.load(open(self.config_path))
        
    def create_model(self):
        tf.reset_default_graph()
        x = tf.placeholder(shape=self.config['input_shape'], dtype=tf.int32, name='input')
        y_ = tf.placeholder(shape=(None, self.config["no_labels"]), dtype=tf.float32, name='label')
        is_training_phase_op = tf.placeholder_with_default(False, shape=[])

        arch = self.config['architecture']

        last_layer = x
        layers = []
        for layer_index, layer_config in enumerate(arch['layers']):
            current_layer = create_layer(layer_config,
                                         last_layer=last_layer,
                                         layer_index=layer_index,
                                         is_training_phase_op=is_training_phase_op)
                                
            layers.append(current_layer)
            last_layer = current_layer

        preds = last_layer
        preds_softmax = tf.nn.softmax(preds)
        loss = tf.losses.softmax_cross_entropy(y_, preds)
        
        if arch["learning_algorithm"] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])

        train_op = optimizer.minimize(loss)

        correct_predictions = tf.equal(tf.argmax(preds_softmax, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # layers.append(tf.get_collection(tf.GraphKeys.VARIABLES, 'conv_0_7_1_1/kernel')[0])

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        summary = tf.summary.merge_all()

        tensors = {
                    "input_op": x,
                    "label_op": y_,
                    "preds_op": preds,
                    "preds_softmax_op": preds_softmax,
                    "accuracy_op": accuracy,
                    "correct_predictions_op": correct_predictions,
                    "layers_op": layers,
                    "is_training_op": is_training_phase_op,
                    "train_op": train_op,
                    "loss_op": loss,
                    "summary_op": summary,
                    }
        print(layers)

        return tensors

    def __get_predictions(self, sess, tensors, dataset_path, string_size, batch_size, encoding_type):
        input_op = tensors['input_op']
        preds_op = tensors['preds_softmax_op']
        loss_op = tensors['loss_op']

        texts, labels, _ = DatasetUtils.read_dataset(dataset_path, return_tuples=False)
        
        new_texts = []
        new_labels = []
        for index, text in enumerate(texts):
            split = split_text(text, string_size, labels[index])
            new_texts += list(map(lambda x: x[0], split))
            new_labels += list(map(lambda x: x[1], split))

        texts = list(map(clean_string, new_texts))
        labels = new_labels

        if encoding_type == 'ONE_HOT':
            texts = list(map(lambda x: get_one_hot_string(x, string_size), texts))
        else:
            texts = list(map(lambda x: get_ascii_encoding(x, string_size), texts))
        
        predictions = []
        for index in tqdm(range(0, len(texts), batch_size)):
            batch = np.expand_dims(np.array(texts[index:index+batch_size]), -1)
            preds = sess.run(preds_op, feed_dict={input_op: batch})
            predictions.append(np.argmax(preds, 1))
        predictions = np.concatenate(predictions)

        labels_values = list(set(labels))
        labels_values.sort()
        
        if len(labels_values) == 2:
            labels_values_dict = {}
            labels_values_dict[labels_values[0]] = 0
            labels_values_dict[labels_values[1]] = 1
        else:
            labels_values_dict = {}
            labels_values_dict[labels_values[0]] = [1,0,0,0,0,0]
            labels_values_dict[labels_values[1]] = [0,1,0,0,0,0]
            labels_values_dict[labels_values[2]] = [0,0,1,0,0,0]
            labels_values_dict[labels_values[3]] = [0,0,0,1,0,0]
            labels_values_dict[labels_values[4]] = [0,0,0,0,1,0]
            labels_values_dict[labels_values[5]] = [0,0,0,0,0,1]
        labels = np.array(list(map(lambda x: labels_values_dict[x], labels)))

        return predictions, labels, labels_values

    def __evaluate(self, sess, tensors, batch_size=50):
        dataset_config = json.load(open(get_absolute_path(self.config["dataset_generator_config_path"])))

        train_dataset_path = get_absolute_path(dataset_config["train_dataset_path"])
        val_dataset_path = get_absolute_path(dataset_config["val_dataset_path"])
        test_dataset_path = get_absolute_path(dataset_config["test_dataset_path"])

        train_preds, train_labels, labels_values = self.__get_predictions(sess, tensors, train_dataset_path, 
                                                    dataset_config["string_size"], batch_size, 
                                                    dataset_config['encoding_type'])

        val_preds, val_labels, labels_values = self.__get_predictions(sess, tensors, val_dataset_path, 
                                                    dataset_config["string_size"], batch_size,
                                                    dataset_config['encoding_type'])
        
        test_preds, test_labels, labels_values = self.__get_predictions(sess, tensors, test_dataset_path, 
                                                    dataset_config["string_size"], batch_size,
                                                    dataset_config['encoding_type'])
        

        train_acc = np.sum(train_preds == train_labels) / train_preds.shape[0]
        val_acc = np.sum(val_preds == val_labels) / val_preds.shape[0]
        test_acc = np.sum(test_preds == test_labels) / test_preds.shape[0]

        print(f"Train Accuracy: {train_acc} Val Accuracy: {val_acc} Test Accuracy: {test_acc}")

        return val_acc, test_acc


    def train(self, tensors=None, load_model_path=None):
        if tensors is None:
            tensors = self.create_model()

        input_op = tensors['input_op']
        label_op = tensors['label_op']
        is_training_op = tensors['is_training_op']
        loss_op = tensors['loss_op']
        train_op = tensors['train_op']
        summary_op = tensors['summary_op']
        preds_op = tensors['preds_op']
        preds_softmax_op = tensors['preds_softmax_op']
        correct_preds_op = tensors['correct_predictions_op']
        layers_op = tensors['layers_op']
        accuracy_op = tensors['accuracy_op']

        dg = DatasetGenerator(self.config["dataset_generator_config_path"])
    
        train_proc, train_queue = dg.get_train_generator()
        train_proc.start()

        val_proc, val_queue = dg.get_validation_generator()
        val_proc.start()

        test_proc, test_queue = dg.get_test_generator()
        test_proc.start()

        tensorboard_train = tf.summary.FileWriter(get_absolute_path(os.path.join(self.config["save_dir"],self.config["tensorboard_path"])))
        tensorboard_train.add_graph(tf.get_default_graph())
        tensorboard_test = tf.summary.FileWriter(get_absolute_path(os.path.join(self.config["save_dir"],self.config["tensorboard_test_path"])))
        tensorboard_validate = tf.summary.FileWriter(get_absolute_path(os.path.join(self.config["save_dir"],self.config["tensorboard_val_path"])))

        tensorboard_acc = tf.summary.FileWriter(get_absolute_path(os.path.join(self.config["save_dir"], "tensorboard_acc")))

        acc_summary = tf.Summary() 
        acc_summary.value.add(tag='val_accuracy', simple_value=None)
        acc_summary.value.add(tag='test_accuracy', simple_value=None)
 
        saver = tf.train.Saver(max_to_keep=100)

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )

        with tf.Session() as sess:
            if load_model_path is None:
                sess.run(tf.global_variables_initializer())
                start_step = 0
            else:
                start_step = int(load_model_path.split('-')[-1]) + 1
                saver.restore(sess, get_absolute_path(load_model_path))

            for i in range(start_step, self.config["steps"]):
                if (i % self.config["validate_step"] != 0):
                    train_batch = train_queue.get()
                    if len(train_batch) == 2:
                        train_data, train_labels = train_batch
                    else:
                        train_data, train_labels, epoch_index = train_batch 
                        print(f"Epoch index: {epoch_index}")

                    _, loss = sess.run([train_op, loss_op],
                            feed_dict = {
                                            input_op: train_data,
                                            label_op: train_labels,
                                            is_training_op: True
                                        })
                else:
                    train_batch = train_queue.get()
                    validate_batch = val_queue.get()
                    test_batch = test_queue.get()
                    
                    train_data = train_batch[0]
                    train_labels = train_batch[1]
                    validate_data = validate_batch[0]
                    validate_labels = validate_batch[1]
                    test_data = test_batch[0]
                    test_labels = test_batch[1]

                    _, loss, summary, accuracy, preds, correct_preds, layers = sess.run([train_op, 
                                                                                loss_op, summary_op,
                                                                                accuracy_op, preds_softmax_op,
                                                                                correct_preds_op, layers_op],
                        feed_dict = {
                                        input_op: train_data,
                                        label_op: train_labels,
                                        is_training_op: True
                                    })

                    vsummary, val_accuracy, val_loss, vpreds, vcorrect_preds = sess.run([summary_op, accuracy_op, loss_op,
                                                                                        preds_softmax_op, correct_preds_op],
                        feed_dict = {
                                        input_op: validate_data,
                                        label_op: validate_labels
                                    })
                    tsummary, test_accuracy, test_loss, tpreds, tcorrect_preds = sess.run([summary_op, accuracy_op, loss_op,
                                                                preds_softmax_op, correct_preds_op],
                        feed_dict = {
                                        input_op: test_data,
                                        label_op: test_labels
                                    })

                    print(f"Train accuracy at step {i} is {accuracy}")
                    print(f"Val accuracy at step {i} is {val_accuracy}")
                    print(f"Test accuracy at step {i} is {test_accuracy}")
                    print(f"train_loss = {loss} val_loss = {val_loss} test_loss = {test_loss}")

                    tensorboard_train.add_summary(summary, i)
                    tensorboard_validate.add_summary(vsummary, i)
                    tensorboard_test.add_summary(tsummary, i)

                if (i % self.config["save_step"] == 0):
                    saver.save(sess, get_absolute_path(os.path.join(self.config["save_dir"], self.config["save_model_name"])), i)
                    if i > 0:
                        acc_valid, acc_test = self.__evaluate(sess, tensors)

                        acc_summary.value[0].simple_value = acc_valid
                        acc_summary.value[1].simple_value = acc_test
                        tensorboard_acc.add_summary(acc_summary, i)

                    

    def save(self):
        pass

    def load(self, model_path, tensors=None):
        if tensors is None:
            tensors = self.create_model()
        
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, get_absolute_path(model_path))
            
            acc_valid, acc_test = self.__evaluate(sess, tensors)

if __name__ == '__main__':
    config_path = "@src/config/classifier_network_config.json"
    a = ClassifierNetwork(config_path)

    tensors = a.create_model()
    a.train(tensors)

    
    # model_path = "@results/models/binary_classifier_1/binary_classifier-99000"
    # a.train(tensors, model_path)
    # a.load(model_path, tensors)
    
