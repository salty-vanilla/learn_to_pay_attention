import argparse
import os
import sys
sys.path.append(os.getcwd())
from cifar.dataset import data_init
from cifar.model import CifarCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', default='cifar10')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=100)
    parser.add_argument('--height', '-ht', type=int, default=32)
    parser.add_argument('--width', '-wd', type=int, default=32)
    parser.add_argument('--save_steps', '-ss', type=int, default=10)
    parser.add_argument('--validation_steps', '-vs', type=int, default=10)
    parser.add_argument('--logdir', '-ld', default='../logs/cifar')

    args = parser.parse_args()

    nb_classes = 10 if args.dataset == 'cifar10' else 100

    (x_train, y_train), (x_val, y_val) = data_init(args.dataset)
    model = CifarCNN((args.width, args.height, 3),
                     nb_classes=nb_classes,
                     logdir=args.logdir)

    model.fit(x_train, y_train,
              x_val, y_val,
              nb_epoch=args.nb_epoch,
              validation_steps=args.validation_steps,
              save_steps=args.save_steps,
              model_dir=args.logdir
              )


if __name__ == '__main__':
    main()
