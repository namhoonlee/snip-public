import os
import sys
import argparse
import tensorflow as tf

from dataset import Dataset
from model import Model
import prune
import train
import test


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data options
    parser.add_argument('--datasource', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./data', help='location to dataset')
    parser.add_argument('--aug_kinds', nargs='+', type=str, default=[], help='augmentations to perform')
    # Model options
    parser.add_argument('--arch', type=str, default='lenet5', help='network architecture to use')
    parser.add_argument('--target_sparsity', type=float, default=0.9, help='level of sparsity to achieve')
    # Train options
    parser.add_argument('--batch_size', type=int, default=100, help='number of examples per mini-batch')
    parser.add_argument('--train_iterations', type=int, default=10000, help='number of training iterations')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer of choice')
    parser.add_argument('--lr_decay_type', type=str, default='constant', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--decay_boundaries', nargs='+', type=int, default=[], help='boundaries for piecewise_constant decay')
    parser.add_argument('--decay_values', nargs='+', type=float, default=[], help='values for piecewise_constant decay')
    # Initialization
    parser.add_argument('--initializer_w_bp', type=str, default='vs', help='initializer for w before pruning')
    parser.add_argument('--initializer_b_bp', type=str, default='zeros', help='initializer for b before pruning')
    parser.add_argument('--initializer_w_ap', type=str, default='vs', help='initializer for w after pruning')
    parser.add_argument('--initializer_b_ap', type=str, default='zeros', help='initializer for b after pruning')
    # Logging, saving, options
    parser.add_argument('--logdir', type=str, default='logs', help='location for summaries and checkpoints')
    parser.add_argument('--check_interval', type=int, default=100, help='check interval during training')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval during training')
    args = parser.parse_args()
    # Add more to args
    args.path_summary = os.path.join(args.logdir, 'summary')
    args.path_model = os.path.join(args.logdir, 'model')
    args.path_assess = os.path.join(args.logdir, 'assess')
    return args


def main():
    args = parse_arguments()

    # Dataset
    dataset = Dataset(**vars(args))

    # Reset the default graph and set a graph-level seed
    tf.reset_default_graph()
    tf.set_random_seed(9)

    # Model
    model = Model(num_classes=dataset.num_classes, **vars(args))
    model.construct_model()

    # Session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # Prune
    prune.prune(args, model, sess, dataset)

    # Train and test
    train.train(args, model, sess, dataset)
    test.test(args, model, sess, dataset)

    sess.close()
    sys.exit()


if __name__ == "__main__":
    main()
