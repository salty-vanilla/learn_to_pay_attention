import tensorflow as tf
from layers import dense, flatten
import os
import time
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils.image import mpl_to_pil
from image_sampler import ImageSampler


class AttentionCNN:
    def __init__(self, image_shape,
                 nb_classes,
                 learning_rate=1e-3,
                 logdir=None):
        self.image_shape = image_shape
        self.image = tf.placeholder(tf.float32,
                                    [None, *image_shape], name='x')
        self.label = tf.placeholder(tf.float32,
                                    [None, nb_classes], name='label')
        self.nb_classes = nb_classes

        self.y_train, _ = self.__call__(self.image)
        self.y_val, self.attentions = self.__call__(self.image,
                                                    is_training=False,
                                                    reuse=True)

        self.attention_shapes = [a.get_shape().as_list()[1:] for a in self.attentions]

        with tf.variable_scope('loss'):
            with tf.variable_scope('train'):
                self.loss_train = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.y_train,
                    labels=self.label
                )
                self.loss_train = tf.reduce_mean(self.loss_train)

            with tf.variable_scope('validation'):
                self.loss_val = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.y_val,
                    labels=self.label
                )
                self.loss_val = tf.reduce_mean(self.loss_val)

        with tf.variable_scope('accuracy'):
            with tf.variable_scope('train'):
                correct_prediction =\
                    tf.equal(tf.argmax(self.y_train, 1),
                             tf.argmax(self.label, 1))
                self.acc_train =\
                    tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            with tf.variable_scope('validation'):
                correct_prediction = \
                    tf.equal(tf.argmax(self.y_val, 1),
                             tf.argmax(self.label, 1))
                self.acc_val = \
                    tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.variable_scope('optimizer'):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                .minimize(self.loss_train)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.logdir = logdir
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            self.summarises = []
            with tf.variable_scope('summary'):
                with tf.variable_scope('train'):
                    self.summary_loss_train = tf.summary.scalar('loss', self.loss_train)
                    self.summary_acc_train = tf.summary.scalar('accuracy', self.acc_train)
            self.tb_writer = tf.summary.FileWriter(logdir,
                                                   graph=self.sess.graph)

    def __call__(self, x,
                 is_training=True,
                 reuse=False,
                 *args,
                 **kwargs):
        raise NotImplementedError

    def fit(self, x,
            y,
            val_x=None,
            val_y=None,
            batch_size=16,
            nb_epoch=100,
            validation_steps=10,
            save_steps=10,
            model_dir='./models'):
        image_sampler = ImageSampler(None,
                                     normalize_mode=None).flow(x, y, batch_size)
        if val_x is not None and val_y is not None:
            assert len(val_x) == len(val_y)
            valid_sampler = ImageSampler(None,
                                         normalize_mode=None,
                                         is_training=False).flow(x, y, batch_size, shuffle=False)
        else:
            valid_sampler = None
        self.fit_generator(image_sampler,
                           valid_sampler,
                           nb_epoch=nb_epoch,
                           validation_steps=validation_steps,
                           save_steps=save_steps,
                           model_dir=model_dir)

    def fit_generator(self, image_sampler,
                      valid_sampler=None,
                      nb_epoch=100,
                      validation_steps=10,
                      save_steps=10,
                      model_dir='./models'):
        os.makedirs(model_dir, exist_ok=True)

        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        global_step = 0
        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                image_batch, label_batch = image_sampler()
                _, loss_train, acc_train, summary_loss_train, summary_acc_train =\
                    self.sess.run([self.opt,
                                   self.loss_train,
                                   self.acc_train,
                                   self.summary_loss_train,
                                   self.summary_acc_train],
                                  feed_dict={self.image: image_batch,
                                             self.label: label_batch})
                print('iter : {} / {}  {:.1f}[s]  loss : {:.4f}  acc : {:.4f}  \r'
                      .format(iter_,
                              steps_per_epoch,
                              time.time() - start,
                              loss_train,
                              acc_train),
                      end='')
                self.tb_writer.add_summary(summary_loss_train, global_step)
                self.tb_writer.add_summary(summary_acc_train, global_step)
                self.tb_writer.flush()
                global_step += 1

            if epoch % validation_steps == 0 and valid_sampler is not None:
                loss_val, acc_val, _ = self.evaluate_generator(valid_sampler)
                self.tb_writer.add_summary(
                    tf.Summary(value=[
                        tf.Summary.Value(tag='summary/validation/loss',
                                         simple_value=loss_val)]),
                    global_step)
                self.tb_writer.add_summary(
                    tf.Summary(value=[
                        tf.Summary.Value(tag='summary/validation/accuracy',
                                         simple_value=acc_val)]),
                    global_step)

                x, y = valid_sampler.random_sampling(4)
                res = self.evaluate_on_batch(x, y)
                atts = res[-1]
                figs = np.array([visualize(ims[0], ims[1:]) for ims in zip(x, *atts)])
                summary = self.sess.run(tf.summary.image('epoch_%d' % epoch,
                                                         figs,
                                                         nb_sample))
                self.tb_writer.add_summary(summary, global_step)

            if epoch % save_steps == 0:
                self.save(model_dir, epoch)

    def save(self, model_dir, epoch):
        dst_dir = os.path.join(model_dir, "epoch_{}".format(epoch))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        return self.saver.save(self.sess, save_path=os.path.join(dst_dir, 'model.ckpt'))
    
    def restore(self, model_path):
        return self.saver.restore(self.sess, model_path)

    def predict(self, x, batch_size=16):
        pred = np.empty([0, 1])
        attentions = [np.empty([0, *shape])
                      for shape in self.attention_shapes]
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o, att = self.predict_on_batch(x_batch)
            pred = np.append(pred, o, axis=0)
            for i in range(len(attentions)):
                attentions[i] = np.append(attentions[i], att[i], axis=0)
        return pred, attentions

    def evaluate(self, x, y, batch_size=16):
        loss = np.empty([0, 1])
        acc = np.empty([0, 1])
        attentions = [np.empty([0, *shape])
                      for shape in self.attention_shapes]
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            y_batch = y[iter_ * batch_size: (iter_ + 1) * batch_size]
            l, a, att = self.evaluate_on_batch(x_batch, y_batch)
            loss = np.append(loss, l * len(x_batch), axis=0)
            acc = np.append(acc, a * len(x_batch), axis=0)
            for i in range(len(attentions)):
                attentions[i] = np.append(attentions[i], att[i], axis=0)
        loss /= len(x)
        acc /= len(x)
        return acc, attentions

    def predict_generator(self, image_sampler):
        pred = np.empty([0, 1])
        attentions = [np.empty([0, *shape])
                      for shape in self.attention_shapes]
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1
        for x_batch in image_sampler():
            if isinstance(x_batch, list):
                x_batch = x_batch[0]
            o, att = self.predict_on_batch(x_batch)
            pred = np.append(pred, o, axis=0)
            for i in range(len(attentions)):
                attentions[i] = np.append(attentions[i], att[i], axis=0)
        return pred, attentions

    def evaluate_generator(self, image_sampler):
        loss = np.empty([0, 1])
        acc = np.empty([0, 1])
        attentions = [np.empty([0, *shape])
                      for shape in self.attention_shapes]
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1
        for x_batch, y_batch in image_sampler():
            l, a, att = self.evaluate_on_batch(x_batch, y_batch)
            loss = np.append(loss, l*len(x_batch))
            acc = np.append(acc, a*len(x_batch))
            for i in range(len(attentions)):
                attentions[i] = np.append(attentions[i], att[i], axis=0)
        loss = sum(loss) / nb_sample
        acc = sum(acc) / nb_sample
        return loss, acc, attentions

    def predict_on_batch(self, x):
        res = self.sess.run([self.y_val, *self.attentions],
                            feed_dict={self.image: x})
        pred = res[0]
        attentions = res[1:]
        return pred, attentions

    def evaluate_on_batch(self, x, y):
        att = self.sess.run(self.attentions,
                            feed_dict={self.image: x,
                                       self.label: y})
        loss, acc = self.sess.run([self.loss_val, self.acc_val],
                                  feed_dict={self.image: x,
                                             self.label: y})
        return loss, acc, att


def visualize(x, attentions, dst_path=None, vmin=0., vmax=1.):
    fig = plt.figure(figsize=(2*(len(attentions)+1), 2))
    plt.subplot(1, len(attentions)+1, 1)
    plt.imshow(x, vmin=0., vmax=1.)
    plt.xticks([])
    plt.yticks([])
    plt.title('input')
    for i, att in enumerate(attentions):
        plt.subplot(1, len(attentions) + 1, i+2)
        x_h, x_w = x.shape[:2]
        att_h, att_w = att.shape[:2]
        att = scipy.ndimage.zoom(att, (x_w/att_w, x_h/att_h), order=1)
        plt.imshow(x, vmin=0., vmax=1.)
        plt.imshow(att, vmin=vmin, vmax=vmax, cmap='jet', alpha=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.title('attention_%d' % i)
    fig.tight_layout()
    if dst_path is not None:
        plt.savefig(dst_path)

    image = mpl_to_pil(fig)
    image = np.asarray(image.convert('RGB'))
    plt.close()
    plt.clf()
    return image


def attention_estimator(x, g):
    with tf.variable_scope(None, 'attention_estimator'):
        _, height, width, x_dim = x.get_shape().as_list()
        g_dim = g.get_shape().as_list()[-1]

        if not x_dim == g_dim:
            x = dense(x, g_dim, use_bias=False)

        c = tf.reduce_sum(x * tf.expand_dims(tf.expand_dims(g, 1), 1), axis=-1)
        a = tf.nn.softmax(flatten(c))
        a = tf.reshape(a, (-1, height, width))
        g_out = x * tf.expand_dims(a, -1)
        g_out = tf.reduce_sum(g_out, axis=[1, 2])
        return g_out, a


def attention_module(ls, g):
    gs = []
    as_ = []
    for l in ls:
        _g, a = attention_estimator(l, g)
        gs.append(_g)
        as_.append(a)
    g = tf.concat(gs, axis=-1)
    return g, as_
