from tensorflow.python.keras.datasets import cifar10, cifar100
from tensorflow.python.keras.utils import to_categorical


def data_init(dataset='cifar10', mode='train', val_rate=0.2):
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    else:
        raise NotImplementedError

    if mode == 'train':
        x_train = x_train.astype('float32') / 255
        y_train = to_categorical(y_train)
        train_index = int((1-val_rate) * len(x_train))
        return (x_train[:train_index], y_train[:train_index]), \
               (x_train[train_index:], y_train[train_index:])
    elif mode == 'test':
        x_test = x_test.astype('float32') / 255
        y_test = to_categorical(y_test)
        return x_test, y_test
