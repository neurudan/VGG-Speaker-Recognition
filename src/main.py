from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
from keras import backend as K
import numpy as np
import wandb
import gc
from wandb.keras import WandbCallback

from generator import DataGenerator
from test_generator import TestDataGenerator
from eval_callback import *

sys.path.append('../tool')
import toolkits


import atexit


@atexit.register
def terminate_subprocesses():
    os.system('pkill -u "neurudan"')

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--batch_size_pretrain', default=64, type=int)
parser.add_argument('--data_path', default='/scratch/local/ssd/weidi/voxceleb2/dev/wav', type=str)
parser.add_argument('--multiprocess', default=12, type=int)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
parser.add_argument('--qsize', default=100000, type=int)
parser.add_argument('--qsize_test', default=10000, type=int)
parser.add_argument('--n_train_proc', default=100, type=int)
parser.add_argument('--n_test_proc', default=32, type=int)
parser.add_argument('--n_speakers', default=1000, type=int)

parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
global args
args = parser.parse_args()

def main():
    # gpu configuration
    import model

    toolkits.initialize_GPU(args)

    network, network_eval = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1),
                                                         num_class=args.n_speakers,
                                                         mode='train', args=args)
    
    print('\n\nGPU Model\n==========================================================================================================')
    for i, layer in enumerate(network.layers):
        layer.trainable = False
        print(f'[{i}]: {layer.name}, {layer}')
    

    print('\n\nReal Model\n==========================================================================================================')
    for i, layer in enumerate(network.layers[-2].layers[-20:]):
        layer.trainable = False
        print(f'[{i}]: {layer.name}, {layer}')

    print('\n\nOutput from VGG\n==========================================================================================================')
    layer = network.layers[-2]
    print(layer.output)
    print(type(layer.output))
    print()
    for out in layer.output:
        print(out)
    
    print('\n\nInput in Merge Layer\n==========================================================================================================')
    layer = network.layers[-1]
    print(layer.input)
    print(type(layer.input))
    print()
    for inp in layer.input:
        print(inp)

    

    



def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    epochs = args.epochs
    stage1, stage2, stage3 = int(epochs * 0.5), int(epochs * 0.8), epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3]
        gamma = [1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    #print('==> Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


def set_path(args):
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    if args.aggregation_mode == 'avg':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    elif args.aggregation_mode == 'vlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_'
                                'bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    elif args.aggregation_mode == 'gvlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_ghost{args.ghost_cluster}_'
                                'bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    else:
        raise IOError('==> unknown aggregation mode.')
    model_path = os.path.join('../model', exp_path)
    log_path = os.path.join('../log', exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


if __name__ == "__main__":
    main()
