import tensorflow as tf
from src.models.models_utils import create_layer
from src.utils.paths_utils import get_absolute_path
from src.dataset.dataset_utils import DatasetUtils
from src.dataset.dataset_generator import split_text, get_ascii_encoding
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import json
from random import shuffle


class EmbeddingNetwork:

    def __init__(self, config_path):
        self.config_path = get_absolute_path(config_path)
        self.config = json.load(open(self.config_path))

    def create_model(self):
        x = tf.placeholder(shape=self.config['input_shape'], dtype=tf.int32, name='input')
        emb = create_layer(["embedding_layer", self.config["alphabet_size"], 
                            self.config["embedding_size"], True], x, layer_index=0)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])

        loss_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        loss = tf.reduce_sum(loss_collection)
        train_op = optimizer.minimize(loss)

        return {
                "input_op": x,
                "emb_op": emb,
                "train_op": train_op,
                "loss_op": loss,
                "loss_collection_op": loss_collection
                }

    def get_dataset(self):
        texts, _, _ = DatasetUtils.read_dataset(self.config["dataset_path"], return_tuples=False)
        
        new_texts = []
        for text in tqdm(texts):
            split = split_text(text, self.config["string_size"])
            new_texts += list(map(lambda x: x[0], split))
        
        pool = Pool(processes=cpu_count())
        partial_fn = partial(get_ascii_encoding, string_size=self.config["string_size"])
        texts = np.array(pool.map(partial_fn, new_texts))

        # texts = np.array(list(map(lambda x: get_ascii_encoding(x, self.config["string_size"]), tqdm(new_texts))))
        texts = np.squeeze(texts)
        print(texts.shape)

        return texts

    def train(self):
        tensors = self.create_model()
        input_op = tensors["input_op"]
        emb_op = tensors["emb_op"]
        train_op = tensors["train_op"]
        loss_op = tensors["loss_op"]
        loss_collection_op = tensors["loss_collection_op"]

        data = self.get_dataset()

        batch_size = self.config["batch_size"]

        print(f'Data shape is {data.shape}')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_index in range(self.config["no_epochs"]):
                data = data[np.random.permutation(data.shape[0])]
                print(f"Epoch: {epoch_index}")
                for i in range(0, data.shape[0], batch_size):
                    for j in range(5):
                        _, loss, emb = sess.run([train_op, loss_op, emb_op], feed_dict = {input_op: data[i:i+batch_size]})
                        import pdb; pdb.set_trace()
                print(np.mean(sess.run(tf.get_default_graph().get_tensor_by_name('EmbedSequence/embeddings:0'))))

            embedding_matrix = sess.run(tf.get_default_graph().get_tensor_by_name('EmbedSequence/embeddings:0'))
            print(f'Saving embedding matrix with shape {embedding_matrix.shape}')
            np.save(get_absolute_path(self.config["emb_matrix_path"]), embedding_matrix) 


if __name__ == "__main__":
    config_path = "@src/config/embedding_config.json"
    
    embedding_network = EmbeddingNetwork(config_path)
    embedding_network.train()