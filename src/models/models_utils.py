import tensorflow as tf
from src.utils.paths_utils import get_absolute_path
import numpy as np



def create_activation_fn(activation_type):
    if activation_type == 'relu':
        return tf.nn.relu
    elif activation_type == 'leaky_relu':
        return tf.nn.leaky_relu
    elif activation_type == 'elu':
        return tf.nn.elu
    elif activation_type == 'tanh':
        return tf.nn.tanh
    elif activation_type == 'sigmoid':
        return tf.nn.sigmoid
    else:
        return None

def create_layer(layer_config, last_layer, layer_index=-1, is_training_phase_op=None):
    layer_type = layer_config[0]

    layer_config_tags = []
    for lc in layer_config:
        if type(lc) is list:
            layer_config_tags.append('_'.join(list(map(str, lc))))
        else:
            layer_config_tags.append(lc)

    if layer_type == 'conv':
        layer_name = f'conv_{layer_index}_{layer_config_tags[1]}_{layer_config_tags[2]}'
        current_layer = tf.layers.conv2d(
                                inputs=last_layer,
                                kernel_size=layer_config[1],
                                filters=layer_config[2],
                                padding='same',
                                activation=create_activation_fn(layer_config[3]),
                                name=layer_name)
    elif layer_type == 'squeeze_and_excitation':
        layer_name = f'squeeze_and_excitation_{layer_index}_{layer_config_tags[1]}'
        no_filters = last_layer.shape[-1]

        layer_1 = tf.reduce_mean(last_layer, axis=[1,2], name=f'{layer_name}_avg_global_pool')
        layer_2 = tf.layers.dense(layer_1, no_filters // layer_config[1], 
                                    activation=tf.nn.relu, name=f'{layer_name}_fc_relu')
        layer_3 = tf.layers.dense(layer_2, no_filters, 
                                    activation=tf.nn.sigmoid, name=f'{layer_name}_fc_sigmoid')
        layer_4 = tf.reshape(layer_3, [-1, 1, 1, no_filters])
        layer_5 = last_layer * layer_4

        return layer_5
    elif layer_type == 'max_pool':
        layer_name = f'max_pool_{layer_index}_{layer_config_tags[1]}'
        current_layer = tf.layers.max_pooling2d(
                                inputs=last_layer,
                                pool_size=layer_config[1],
                                padding='same',
                                strides=layer_config[2],
                                name=layer_name)
    elif layer_type == 'avg_pool':
        layer_name = f'avg_pool_{layer_index}_{layer_config_tags[1]}'
        current_layer = tf.layers.average_pooling2d(
                                inputs=last_layer,
                                pool_size=layer_config[1],
                                strides=layer_config[2],
                                name=layer_name)
    elif layer_type == 'avg_global_pool':
        layer_name = f'avg_global_pool_{layer_index}'
        current_layer = tf.reduce_mean(last_layer, axis=[1,2], name=layer_name)
    elif layer_type == 'flatten':
        layer_name = f'flatten_{layer_index}'
        current_layer = tf.layers.flatten(
                                inputs=last_layer,
                                name=layer_name)
    elif layer_type == 'dense':
        layer_name = f'dense_{layer_index}_{layer_config_tags[1]}'
        current_layer = tf.layers.dense(
                                inputs=last_layer,
                                units=layer_config[1],
                                activation=create_activation_fn(layer_config[2]),
                                name=layer_name)
    elif layer_type == 'dropout':
        layer_name = f'dropout_{layer_index}_{layer_config_tags[1]}'
        current_layer = tf.layers.dropout(last_layer, 
                                          rate=layer_config[1],
                                          training=is_training_phase_op)
    elif layer_type == 'embedding_dense':
        layer_name = 'embedding_dense_layer'
        current_layer = tf.layers.dense(
                                inputs=last_layer,
                                units=layer_config[1],
                                activation=create_activation_fn(layer_config[2]),
                                name=layer_name)
    elif layer_type == 'reshape':
        layer_name = f'reshape_{layer_index}'
        current_layer = tf.reshape(last_layer, layer_config[1])
    elif layer_type == 'embedding_layer':
        layer_name = f'embedding_layer_{layer_index}'
        current_layer = tf.contrib.layers.embed_sequence(last_layer, 
                                            vocab_size=layer_config[1],
                                            embed_dim=layer_config[2])
    elif layer_type == f'mean_layer':
        layer_name = f'mean_layer_{layer_index}'
        current_layer = tf.reduce_mean(last_layer, 
                                axis=layer_config[1],
                                name=layer_name)

    return current_layer





# anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3, flags.embedding_size]), 3, 1)
#         pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
#         neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
#         basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        
        
#         if flags.loss_fn == 'max':
#             loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
#         else:
#             loss = tf.reduce_mean(tf.exp(basic_loss))