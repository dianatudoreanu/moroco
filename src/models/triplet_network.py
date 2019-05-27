from src.models.abstract_neural_network import AbstractNeuralNetwork
from src.models.models_utils import create_layer
from src.utils.paths_utils import get_absolute_path
from src.dataset.dataset_generator import DatasetGenerator, get_one_hot_string, split_text, decode_one_hot, get_ascii_encoding
from src.dataset.dataset_utils import DatasetUtils
from src.utils.string_utils import clean_string
from sklearn.svm import SVC
from tqdm import tqdm
import tensorflow as tf
import json
import numpy as np
import os

class TripletNetwork(AbstractNeuralNetwork):

    def __init__(self, config_path):
        self.config_path = get_absolute_path(config_path)
        self.config = json.load(open(self.config_path))
        self.dataset_config = json.load(open(get_absolute_path(self.config['dataset_generator_config_path'])))
        
    def create_model(self):
        tf.reset_default_graph()
        x = tf.placeholder(shape=self.config['input_shape'], dtype=tf.int32, name='input')
        print(x)
        is_training_phase_op = tf.placeholder_with_default(False, shape=[])

        arch = self.config['architecture']

        last_layer = x
        layers = []
        for layer_index, layer_config in enumerate(arch['layers']):
            layer_type = layer_config[0]
            
            current_layer = create_layer(layer_config,
                                         last_layer=last_layer,
                                         layer_index=layer_index,
                                         is_training_phase_op=is_training_phase_op)
            if layer_type == 'embedding_dense':
                embedding_layer = current_layer
                embedding_size = layer_config[1]
                                
            layers.append(current_layer)
            last_layer = current_layer

        num = tf.reshape(tf.linalg.norm(embedding_layer, axis=1), (-1, 1))
        mask = tf.to_float(tf.equal(num, 0.0))
        num = num + mask * 1

        embedding_layer = embedding_layer / num
        anchor, pos, neg = tf.unstack(tf.reshape(embedding_layer, [-1, 3, embedding_size]), 3, 1)
        
        pos_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), 1))
        neg_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), 1))

        basic_loss = pos_dist - neg_dist + arch['alpha']
        loss = tf.reduce_mean(tf.minimum(tf.maximum(basic_loss, 0.0), 2.0))

        #BPR (Bayesian Personalized Ranking) loss
        # loss = tf.reduce_mean(1.0 - tf.nn.sigmoid(pos_dist - neg_dist))

        if arch["learning_algorithm"] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])

        train_op = optimizer.minimize(loss)

        tf.summary.scalar("loss", loss)
        summary = tf.summary.merge_all()

        tensors = {
                    "input_op": x,
                    "embedding_op": embedding_layer,
                    "pos_dist_op": pos_dist,
                    "neg_dist_op": neg_dist,
                    "layers_op": layers,
                    "is_training_op": is_training_phase_op,
                    "train_op": train_op,
                    "loss_op": loss,
                    "summary_op": summary,
                    }
        print(layers)
        
        return tensors

    def __get_embeddings(self, sess, tensors, dataset_path, string_size, batch_size, encoding_type):
        input_op = tensors['input_op']
        embedding_op = tensors['embedding_op']

        texts, labels, _ = DatasetUtils.read_dataset(dataset_path, return_tuples=False)
        
        new_texts = []
        new_labels = []
        for index, text in enumerate(texts):
            split = split_text(text, string_size, labels[index])
            new_texts += list(map(lambda x: x[0], split))
            new_labels += list(map(lambda x: x[1], split))

        texts = new_texts
        labels = new_labels
        
        if encoding_type == 'ONE_HOT':
            texts = list(map(lambda x: get_one_hot_string(clean_string(x), string_size), texts))
        else:
            texts = list(map(lambda x: get_ascii_encoding(clean_string(x), string_size), texts))
        
        emb = []
        for index in tqdm(range(0, len(texts), batch_size)):
            batch = np.expand_dims(np.array(texts[index:index+batch_size]), -1)
            emb.append(sess.run(embedding_op, feed_dict={input_op: batch}))
        emb = np.concatenate(emb)

        normemb = emb

        labels_values = list(set(labels))
        labels_values.sort()
        
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

        return normemb, labels, labels_values

    def __random_sample(self, embeddings, random_size=500):
        return embeddings[np.random.permutation(embeddings.shape[0])][:random_size]

    def __get_embeddings_distances(self, embs, random_sample):
        distances = []

        for emb in embs:
            distances.append([np.sqrt(np.sum(np.square(emb - sample))) for sample in random_sample])

        return np.array(distances)

    def __test_svm(self, sess, tensors, batch_size=50):
        dataset_config = json.load(open(get_absolute_path(self.config["dataset_generator_config_path"])))

        train_dataset_path = get_absolute_path(dataset_config["train_dataset_path"])
        val_dataset_path = get_absolute_path(dataset_config["val_dataset_path"])
        test_dataset_path = get_absolute_path(dataset_config["test_dataset_path"])

        train_emb, train_labels, labels_values = self.__get_embeddings(sess, tensors, train_dataset_path, 
                                                    dataset_config["string_size"], batch_size, 
                                                    dataset_config['encoding_type'])

        val_emb, val_labels, labels_values = self.__get_embeddings(sess, tensors, val_dataset_path, 
                                                    dataset_config["string_size"], batch_size,
                                                    dataset_config['encoding_type'])
        
        test_emb, test_labels, labels_values = self.__get_embeddings(sess, tensors, test_dataset_path, 
                                                    dataset_config["string_size"], batch_size,
                                                    dataset_config['encoding_type'])

        random_sample = self.__random_sample(train_emb)

        def distance(i1, i2):
            print(np.sqrt(np.sum(np.square(train_emb[i1] - train_emb[i2]))), train_labels[i1], train_labels[i2])

        # train_emb = self.__get_embeddings_distances(train_emb, random_sample)
        # val_emb = self.__get_embeddings_distances(val_emb, random_sample)
        # test_emb = self.__get_embeddings_distances(test_emb, random_sample)

        print(np.sum(train_emb < 0.8), np.sum(train_emb > 0.8))
        print(np.sum(val_emb < 0.8), np.sum(val_emb > 0.8))
        print(np.sum(test_emb < 0.8), np.sum(test_emb > 0.8))


        model = SVC(kernel='rbf', C=1.0)
        model.fit(train_emb, train_labels)

        train_preds = model.predict(train_emb)
        train_acc = np.sum(train_preds == train_labels) / train_labels.shape[0]

        val_preds = model.predict(val_emb)
        val_acc = np.sum(val_preds == val_labels) / val_labels.shape[0]

        test_preds = model.predict(test_emb)
        test_acc = np.sum(test_preds == test_labels) / test_labels.shape[0]
        
        # import pdb; pdb.set_trace()

        print(f"Train accuracy: {train_acc} Val Accuracy: {val_acc} Test Accuracy: {test_acc}")

        return val_acc, test_acc


    def train(self, tensors=None, model_path=None):
        if tensors is None:
            tensors = self.create_model()

        input_op = tensors['input_op']
        embedding_op = tensors['embedding_op']
        is_training_op = tensors['is_training_op']
        loss_op = tensors['loss_op']
        train_op = tensors['train_op']
        summary_op = tensors['summary_op']
        pos_dist_op = tensors['pos_dist_op']
        neg_dist_op = tensors['neg_dist_op']
        layers_op = tensors['layers_op']

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

        def normalize_labels(labels):
            if self.dataset_config['label_type'] == 'SIAMESE_STRING_KERNEL':
                return 1.0 - labels
            else:
                return labels

        with tf.Session() as sess:
            if model_path is None:
                sess.run(tf.global_variables_initializer())
                start_index = 0
            else:
                saver.restore(sess, get_absolute_path(model_path))
                start_index = int(model_path.split('-')[-1]) + 1

            for i in range(start_index, self.config["steps"]):
                if (i % self.config["validate_step"] != 0):
                    train_batch = train_queue.get()
                    if len(train_batch) == 2:
                        train_data, train_labels = train_batch
                    else:
                        train_data, train_labels, epoch_index = train_batch
                        print(f"Epoch index: {epoch_index}")

                    train_labels = normalize_labels(train_labels)

                    _, loss = sess.run([train_op, loss_op],
                            feed_dict = {
                                            input_op: train_data,
                                            is_training_op: True
                                        })
                    
                    # def dist(x, y):
                    #     return np.sqrt(np.sum(np.square(x - y)))
                else:
                    train_batch = train_queue.get()
                    validate_batch = val_queue.get()
                    test_batch = test_queue.get()
                    
                    train_data = train_batch[0]
                    train_labels = normalize_labels(train_batch[1])
                    validate_data = validate_batch[0]
                    validate_labels = normalize_labels(validate_batch[1])
                    test_data = test_batch[0]
                    test_labels = normalize_labels(test_batch[1])

                    _, loss, summary, pos_dist, neg_dist, emb = sess.run([train_op, loss_op, 
                                                            summary_op, pos_dist_op,
                                                            neg_dist_op, embedding_op],
                            feed_dict = {
                                            input_op: train_data,
                                            is_training_op: True
                                        })
                    print(pos_dist)
                    print(neg_dist)
                    print(train_labels)

                    # if  i % 1000 == 0:
                    #     import pdb; pdb.set_trace()
                    vloss, vsummary = sess.run([loss_op, summary_op],
                        feed_dict = {
                                        input_op: validate_data
                                    })

                    tloss, tsummary = sess.run([loss_op, summary_op],
                        feed_dict = {
                                        input_op: test_data
                                    })

                    print(f'#{i} Train loss is {loss}')
                    print(f'#{i} Val loss is {vloss}')
                    print(f'#{i} Test loss is {tloss}')
                    
                    tensorboard_train.add_summary(summary, i)
                    tensorboard_validate.add_summary(vsummary, i)
                    tensorboard_test.add_summary(tsummary, i)

                if (i % self.config["save_step"] == 0):
                    saver.save(sess, get_absolute_path(os.path.join(self.config["save_dir"], self.config["save_model_name"])), i)
                    if i > 0:
                        acc_valid, acc_test = self.__test_svm(sess, tensors)

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
            
            acc_valid, acc_test = self.__test_svm(sess, tensors)

if __name__ == '__main__':
    config_path = "@src/config/triplet_loss_config.json"
    a = TripletNetwork(config_path)

    tensors = a.create_model()

    model_path = "@results/models/triplet_loss_2/triplet_loss-10000"

    a.train(tensors)

    # a.load(model_path, tensors)
    
