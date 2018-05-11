import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl
import numpy as np


def dense(x, units, activation_=None):
    return activation(kl.Dense(units, activation=None)(x),
                      activation_)


def conv1d(x, filters,
           kernel_size=3,
           stride=1,
           padding='same',
           activation_=None,
           is_training=True):
    """
    Args:
        x: input_tensor (N, L)
        filters: number of filters
        kernel_size: int
        stride: int
        padding: 'same' or 'valid'
        activation_: activation function
        is_training: True or False
    Returns:
    """
    with tf.variable_scope(None, conv1d.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = activation(kl.Conv2D(filters, (kernel_size, 1), (stride, 1), padding,
                                  activation=None, trainable=is_training)(_x),
                        activation_)
        _x = tf.squeeze(_x, axis=2)
    return _x


def conv1d_transpose(x, filters,
                     kernel_size=3,
                     stride=2,
                     padding='same',
                     activation_=None,
                     is_training=True):
    """
    Args:
        x: input_tensor (N, L)
        filters: number of filters
        kernel_size: int
        stride: int
        padding: 'same' or 'valid'
        activation_: activation function
        is_training: True or False
    Returns: tensor (N, L_)
    """
    with tf.variable_scope(None, conv1d_transpose.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = activation(kl.Conv2DTranspose(filters, (kernel_size, 1), (stride, 1), padding,
                                           activation=None, trainable=is_training)(_x),
                        activation_)
        _x = tf.squeeze(_x, axis=2)
    return _x


def max_pool1d(x, kernel_size=2, stride=2, padding='same'):
    """
    Args:
        x: input_tensor (N, L)
        kernel_size: int
        stride: int
        padding: 'same' or 'valid'
    Returns: tensor (N, L//ks)
    """
    with tf.name_scope(max_pool1d.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = kl.MaxPool2D((kernel_size, 1), (stride, 1), padding)(_x)
        _x = tf.squeeze(_x, axis=2)
    return _x


def average_pool1d(x, kernel_size=2, stride=2, padding='same'):
    """
    Args:
        x: input_tensor (N, L)
        kernel_size: int
        stride: int
        padding: same
    Returns: tensor (N, L//ks)
    """
    _x = tf.expand_dims(x, axis=2)
    _x = kl.AveragePooling2D((kernel_size, 1), (stride, 1), padding)(_x)
    _x = tf.squeeze(_x, axis=2)
    return _x


def upsampling1d(x, size=2):
    """
    Args:
        x: input_tensor (N, L)
        size: int
    Returns: tensor (N, L*ks)
    """
    _x = tf.expand_dims(x, axis=2)
    _x = kl.UpSampling2D((size, 1))(_x)
    _x = tf.squeeze(_x, axis=2)
    return _x


def conv2d(x, filters,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation_: str =None,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros',
           kernel_regularizer=None,
           bias_regularizer=None,
           is_training=True):
    return activation(kl.Conv2D(filters,
                                kernel_size,
                                strides,
                                padding,
                                activation=None,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                trainable=is_training)(x),
                      activation_)


def conv2d_transpose(x, filters,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     padding='same',
                     activation_=None,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     is_training=True):
    return activation(kl.Conv2DTranspose(filters,
                                         kernel_size,
                                         strides,
                                         padding,
                                         activation=None,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         trainable=is_training)(x),
                      activation_)


def subpixel_conv2d(x, filters,
                    rate=2,
                    kernel_size=(3, 3),
                    activation_: str = None,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    is_training=True):
    with tf.variable_scope(None, subpixel_conv2d.__name__):
        _x = conv2d(x, filters*(rate**2),
                    kernel_size,
                    strides=(1, 1),
                    activation_=activation_,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    is_training=is_training)
        _x = pixel_shuffle(_x)
    return _x


def pixel_shuffle(x, r=2):
    with tf.name_scope(pixel_shuffle.__name__):
        return tf.depth_to_space(x, r)


def reshape(x, target_shape):
    return kl.Reshape(target_shape)(x)


def activation(x, func=None):
    if func == 'lrelu':
        return kl.LeakyReLU(0.2)(x)
    elif func == 'swish':
        return x * kl.Activation('sigmoid')(x)
    else:
        return kl.Activation(func)(x)


def layer_norm(x, is_training=True):
    return tl.layer_norm(x, trainable=is_training)


def flatten(x):
    return kl.Flatten()(x)


def global_average_pool2d(x):
    return tf.reduce_mean(x, axis=[1, 2], name=global_average_pool2d.__name__)


def dropout(x,
            rate=0.5,
            is_training=True):
    return tl.dropout(x, 1.-rate,
                      is_training=is_training)


def average_pool2d(x,
                   kernel_size=(2, 2),
                   strides=(2, 2),
                   padding='same'):
    return kl.AveragePooling2D(kernel_size, strides, padding)(x)


def max_pool2d(x,
               kernel_size=(2, 2),
               strides=(2, 2),
               padding='same'):
    return kl.MaxPool2D(kernel_size, strides, padding)(x)


def batch_norm(x, is_training=True):
    return tl.batch_norm(x,
                         scale=True,
                         updates_collections=None,
                         is_training=is_training)
