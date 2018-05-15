import numpy as np
import os
import pandas as pd
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_train', '-lt')
    parser.add_argument('--loss_val', '-lv')
    parser.add_argument('--acc_train', '-at')
    parser.add_argument('--acc_val', '-av')

    args = parser.parse_args()

    loss_train = pd.read_csv(args.loss_train)
    loss_val = pd.read_csv(args.loss_val)
    acc_train = pd.read_csv(args.acc_train)
    acc_val = pd.read_csv(args.acc_val)

    dst_dir = os.path.dirname(args.loss_train)

    # loss
    plt.figure()
    plt.grid()
    plt.plot(loss_train['Step'], loss_train['Value'], label='train')
    plt.plot(loss_val['Step'], loss_val['Value'], label='val')
    plt.xlim(0., loss_val['Step'].max())
    plt.ylim(0., round(np.maximum(loss_val['Value'].max(),
                                  loss_train['Value'].max()) + 0.5))
    plt.xlabel('Step')
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.1),
               ncol=2)
    plt.savefig(os.path.join(dst_dir, 'loss.png'), dpi=300, bbox_inches='tight')

    # acc
    plt.figure()
    plt.grid()
    plt.plot(acc_train['Step'], acc_train['Value'], label='train')
    plt.plot(acc_val['Step'], acc_val['Value'], label='val')
    plt.xlim(0., acc_val['Step'].max())
    plt.ylim(0., 1.)
    plt.xlabel('Step')
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.1),
               ncol=2)
    plt.savefig(os.path.join(dst_dir, 'acc.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()