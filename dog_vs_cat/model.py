import tensorflow as tf
from attention_cnn import AttentionCNN, attention_module
from blocks import conv_block
from layers import flatten, dense


class CifarCNN(AttentionCNN):
    def __call__(self, x,
                 is_training=True,
                 reuse=False,
                 *args,
                 **kwargs):
        with tf.variable_scope(self.__class__.__name__) as vs:
            if reuse:
                vs.reuse_variables()
            conv_params = {'is_training': is_training,
                           'activation_': 'relu',
                           'normalization': 'batch'}
            x = conv_block(x, 64, **conv_params, dropout_rate=0.3)
            x = conv_block(x, 64, **conv_params, dropout_rate=0.3)

            x = conv_block(x, 128, **conv_params, dropout_rate=0.4)
            x = conv_block(x, 128, **conv_params, dropout_rate=0.4)

            x = conv_block(x, 256, **conv_params, dropout_rate=0.4)
            x = conv_block(x, 256, **conv_params, dropout_rate=0.4)
            x = conv_block(x, 256, **conv_params, sampling='pool')
            l1 = x

            x = conv_block(x, 512, **conv_params, dropout_rate=0.4)
            x = conv_block(x, 512, **conv_params, dropout_rate=0.4)
            x = conv_block(x, 512, **conv_params, sampling='pool')
            l2 = x

            x = conv_block(x, 512, **conv_params, dropout_rate=0.4)
            x = conv_block(x, 512, **conv_params, dropout_rate=0.4)
            x = conv_block(x, 512, **conv_params, sampling='pool')
            l3 = x

            x = conv_block(x, 512, **conv_params, sampling='pool')
            x = conv_block(x, 512, **conv_params, sampling='pool')

            x = flatten(x)
            g = dense(x, 512, activation_='relu')

            x, attentions = attention_module([l1, l2, l3], g)
            x = dense(x, self.nb_classes)
            return x, attentions
