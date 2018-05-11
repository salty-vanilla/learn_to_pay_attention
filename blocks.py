from layers import *


def conv_block(x,
               filters,
               activation_,
               kernel_size=(3, 3),
               sampling='same',
               normalization=None,
               is_training=True,
               dropout_rate=0.0,
               mode='conv_first'):
    assert mode in ['conv_first', 'normalization_first']
    assert sampling in ['deconv', 'subpixel', 'stride', 'same', 'pool']
    assert normalization in ['batch', 'layer', None]

    conv_func = conv2d_transpose if sampling == 'deconv' \
        else subpixel_conv2d if sampling == 'subpixel'\
        else conv2d
    normalize = batch_norm if normalization == 'batch' \
        else layer_norm if normalization == 'layer' \
        else None
    strides = (1, 1) if sampling in ['same', 'subpixel'] else (2, 2)

    with tf.variable_scope(None, conv_block.__name__):
        if mode == 'conv_first':
            _x = conv_func(x,
                           filters,
                           kernel_size=kernel_size,
                           activation_=None,
                           strides=strides)

            if normalize is not None:
                _x = normalize(_x, is_training)

            _x = activation(_x, activation_)

            if dropout_rate != 0:
                _x = dropout(_x, dropout_rate)

        else:
            if normalization is None:
                raise ValueError
            else:
                _x = normalize(x, is_training)

            _x = activation(_x, activation_)
            _x = conv_func(_x,
                           filters,
                           kernel_size=kernel_size,
                           activation_=None,
                           strides=strides)
        if sampling == 'pool':
            _x = max_pool2d(x)
        return _x


def residual_block(x,
                   filters,
                   activation_,
                   kernel_size=(3, 3),
                   sampling='same',
                   normalization=None,
                   is_training=True,
                   dropout_rate=0.0,
                   mode='conv_first'):
    with tf.variable_scope(None, residual_block.__name__):
        _x = conv_block(x,
                        filters=filters,
                        activation_=activation_,
                        kernel_size=kernel_size,
                        sampling='same',
                        normalization=normalization,
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        mode=mode)
        _x = conv_block(_x,
                        filters=filters,
                        activation_=None,
                        kernel_size=kernel_size,
                        sampling=sampling,
                        normalization=normalization,
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        mode=mode)

        if x.get_shape().as_list()[-1] != filters:
            __x = conv_block(_x,
                             filters=filters,
                             activation_=None,
                             kernel_size=kernel_size,
                             sampling=sampling,
                             normalization=normalization,
                             is_training=is_training,
                             dropout_rate=dropout_rate,
                             mode=mode)
        elif sampling != 'same':
            __x = conv_block(_x,
                             filters=filters,
                             activation_=None,
                             kernel_size=kernel_size,
                             sampling=sampling,
                             normalization=normalization,
                             is_training=is_training,
                             dropout_rate=dropout_rate,
                             mode=mode)
        else:
            __x = x
        return _x + __x
