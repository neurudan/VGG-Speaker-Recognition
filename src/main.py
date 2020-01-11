from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
import numpy as np
import wandb
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
    wandb.init()

    verify_normal = load_verify_list('../meta/voxceleb1_veri_test.txt')
    verify_hard = load_verify_list('../meta/voxceleb1_veri_test_hard.txt')
    verify_extended = load_verify_list('../meta/voxceleb1_veri_test_extended.txt')

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    import generator

    # construct the data generator.
    params = {'spec_len': 250,
              'batch_size': args.batch_size,
              'normalize': True,
              'qsize': args.qsize,
              'n_proc': args.n_train_proc,
              'n_speakers': args.n_speakers
              }

    # Generators
    print()
    print('==> Initialize Data Generators')
    print()
    trn_gen = DataGenerator(**params)
    eval_cb = EvalCallback(args.n_test_proc, args.qsize_test, params['normalize'])
    print()
    print()

    network, network_eval = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1),
                                                         num_class=args.n_speakers,
                                                         mode='train', args=args)

    eval_cb.model_eval = network_eval

    # ==> load pre-trained model ???list_IDs_temp
    mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())
    if args.resume:
        if os.path.isfile(args.resume):
            if mgpu == 1: network.load_weights(os.path.join(args.resume))
            else: network.layers[mgpu + 1].load_weights(os.path.join(args.resume))
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))
    print(network.summary())
    print('==> gpu {} is, training using loss: {}, aggregation: {}'.format(args.gpu, args.loss, args.aggregation_mode))

    model_path, _ = set_path(args)
    
    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    save_best = keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{acc:.3f}.h5'),
                                                monitor='loss', mode='min', save_best_only=True)

    callbacks = [save_best, normal_lr]

    initial_epoch = True

    for epoch in range(int(args.epochs / 2)):
        pre_acc = 0.0
        pre_loss = 8.0
        if initial_epoch:
            
            optimizer_backup = network.optimizer

            # make all layers except the last and first (input layer) one untrainable
            for layer in network.layers[1:-1]:
                layer.trainable = False

            network.compile(optimizer=keras.optimizers.Adam(lr=step_decay(epoch*2)), 
                            loss='categorical_crossentropy', 
                            metrics=['acc'])

            print("==> starting pretrain phase")
            h = network.fit_generator(trn_gen,
                                      steps_per_epoch=trn_gen.steps_per_epoch,
                                      epochs=2,
                                      verbose=1)
            pre_acc = np.mean(h.history['acc'])
            pre_loss = np.mean(h.history['loss'])
            
            for layer in network.layers[1:-1]:
                layer.trainable = True

            network.compile(optimizer=optimizer_backup, 
                            loss='categorical_crossentropy', 
                            metrics=['acc'])
        

        print("==> starting training phase")
        h = network.fit_generator(trn_gen,
                                  steps_per_epoch=trn_gen.steps_per_epoch,
                                  epochs=(epoch+1) * 2,
                                  initial_epoch=epoch * 2,
                                  callbacks=callbacks,
                                  verbose=1)

        trn_gen.redraw_speakers()
        embeddings = generate_embeddings(eval_cb.model_eval, eval_cb.test_generator)
        eer = calculate_eer(eval_cb.full_list, embeddings)
        wandb.log({'EER': eer,
                   'acc': np.mean(h.history['acc']),
                   'loss': np.mean(h.history['loss']),
                   'lr': step_decay(epoch * 2),
                   'pre_acc': pre_acc,
                   'pre_loss': pre_loss})
        initial_epoch = False


    unique_list = create_unique_list([verify_normal, verify_hard, verify_extended])

    test_generator = eval_cb.test_generator
    test_generator.build_index_list(unique_list)
    
    embeddings = generate_embeddings(network_eval, test_generator)
    
    eer_normal = calculate_eer(verify_normal, embeddings)
    eer_hard = calculate_eer(verify_hard, embeddings)
    eer_extended = calculate_eer(verify_extended, embeddings)

    wandb.log({'EER_normal': eer_normal,
               'EER_hard': eer_hard,
               'EER_extended': eer_extended})

    test_generator.terminate()
    trn_gen.terminate()



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
