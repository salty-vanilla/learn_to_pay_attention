import argparse
from image_sampler import ImageSampler
from models import PaperVGG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir')
    parser.add_argument('val_dir')
    parser.add_argument('--nb_classes', '-nc', type=int, default=2)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=100)
    parser.add_argument('--height', '-ht', type=int, default=32)
    parser.add_argument('--width', '-wd', type=int, default=32)
    parser.add_argument('--save_steps', '-ss', type=int, default=10)
    parser.add_argument('--validation_steps', '-vs', type=int, default=10)
    parser.add_argument('--logdir', '-ld', default='../logs')

    args = parser.parse_args()

    image_sampler = ImageSampler(target_size=(args.width, args.height),
                                 color_mode='rgb',
                                 normalize_mode='sigmoid')

    val_sampler = ImageSampler(target_size=(args.width, args.height),
                               color_mode='rgb',
                               normalize_mode='sigmoid',
                               is_training=False)

    model = PaperVGG((args.width, args.height, 3),
                     nb_classes=args.nb_classes,
                     logdir=args.logdir)

    model.fit_generator(image_sampler.flow_from_directory(args.train_dir,
                                                          batch_size=args.batch_size,
                                                          with_class=True),
                        val_sampler.flow_from_directory(args.val_dir,
                                                        batch_size=args.batch_size,
                                                        with_class=True,
                                                        shuffle=False),
                        nb_epoch=args.nb_epoch,
                        validation_steps=args.validation_steps,
                        save_steps=args.save_steps,
                        model_dir=args.logdir)


if __name__ == '__main__':
    main()
