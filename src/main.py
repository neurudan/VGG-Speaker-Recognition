from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
from keras import backend as K
import numpy as np
import wandb
import gc
import time
from wandb.keras import WandbCallback

from generator import DataGenerator
from test_generator import TestDataGenerator
from eval_callback import *
from multiprocessing import Process, Queue

sys.path.append('../tool')
import toolkits
import json


import atexit


@atexit.register
def terminate_subprocesses():
    os.system('pkill -u "neurudan"')

def clear_queue(queue):
    dat = []
    try:
        while True:
            dat.append(queue.get(timeout=0.5))
    except:
        pass
    m1, m2, u1, u2 = zip(*dat)
    d, nd = [m1, m2, u1, u2], []
    for v in d:
        nd.append(np.mean(list(v)))
    return nd

def gpu_logger(queue):
    import subprocess
    while True:
        lines = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8').split('\n')
        g1, g2 = lines[8], lines[11]
        mem_g1 = float(g1.split('|')[2].split('M')[0].strip())
        mem_g2 = float(g2.split('|')[2].split('M')[0].strip())
        usg_g1 = float(g1.split('|')[3].split('%')[0].strip())
        usg_g2 = float(g2.split('|')[3].split('%')[0].strip())
        queue.put((mem_g1, mem_g2, usg_g1, usg_g2))
        time.sleep(5)

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
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
parser.add_argument('--num_train_ep', default=2, type=int)
parser.add_argument('--num_pretrain_ep', default=2, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--batch_size_pretrain', default=64, type=int)

parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
global args
args = parser.parse_args()


def save_log(eer, lr, best, initial, 
             h_t, h_p, 
             g_t, g_p, 
             t_t, t_p, t_h):

    t_t, t_p, t_h = t_t / 60.0, t_p / 60.0, t_h / 60.0
    b = np.minimum(eer, best['EER'])
    best['EER'] = b
    log = {'EER': eer, 'EER Best': b, 'lr': lr}
    if not initial:
        b = np.minimum(t_h, best['time'])
        best['time'] = b
        log['Hyperepoch - time needed'] = t_h
        log['Hyperepoch - time needed best'] = b
    h = {'train': h_t, 'pretrain': h_p}
    g = {'train': g_t, 'pretrain': g_p}
    t = {'train': t_t, 'pretrain': t_p}
    f = {'acc': [np.maximum, np.max], 'loss': [np.minimum, np.min]}
    for mode in ['train', 'pretrain']:
        if h[mode] is not None:
            for k in ['acc', 'loss']:
                for i in range(len(h[mode][k][:-1])):
                    log[f'{mode} - {k}: {i+1}. epoch'] = h[mode][k][i]
                
                log[f'{mode} - {k}: final epoch'] = h[mode][k][-1]
                log[f'{mode} - {k}: mean'] = np.mean(h[mode][k])
                b = f[k][1](f[k][0](h[mode][k], best[mode][k]))
                best[mode][k] = b
                log[f'{mode} - {k}: best'] = b
        for i, k in enumerate(['GPU 1: Memory', 'GPU 2: Memory', 'GPU 1: Usage', 'GPU 2: Usage']):
            log[f'{mode} - {k}'] = g[mode][i]
        log[f'{mode} - time needed'] = t[mode]
        b = np.minimum(t[mode], best[mode]['time'])
        best[mode]['time'] = b
        log[f'{mode} - time needed best'] = b
    wandb.log(log)
    return best


def main():
    best = {'EER': 1.0,
            'time': 1000000000.0,
            'pretrain': {'acc': 0.0, 'loss': 1000000000.0, 'time': 1000000000.0},
            'train': {'acc': 0.0, 'loss': 1000000000.0, 'time': 1000000000.0}}
    config = {}
    if args.resume:
        with open('previous_run.json', 'r') as fp:
            data = json.load(fp)
            run_id = data['run_id']
            last_epoch = data['last_epoch']
            best = data['best']

            args.epochs = data['epochs']
            args.lr = data['lr']
            args.warmup_ratio = data['warmup_ratio']
            args.loss = data['loss']
            args.optimizer = data['optimizer']
            args.qsize = data['qsize_train']
            args.qsize_test = data['qsize_test']
            args.n_train_proc = data['n_train_proc']
            args.n_test_proc = data['n_test_proc']
            args.n_speakers = data['n_speakers']
            args.num_train_ep = data['num_train_ep']
            args.num_pretrain_ep = data['num_pretrain_ep']
            args.batch_size = data['batch_size_train']
            args.batch_size_pretrain = data['batch_size_pretrain']
            args.bottleneck_dim = data['bottleneck_dim']

            wandb.init(resume=True, id=run_id)
            config = data
    else:
        config = {'epochs': args.epochs,
                  'lr': args.lr,
                  'warmup_ratio': args.warmup_ratio,
                  'loss': args.loss,
                  'optimizer': args.optimizer,
                  'qsize_train': args.qsize,
                  'qsize_test': args.qsize_test,
                  'n_train_proc': args.n_train_proc,
                  'n_test_proc': args.n_test_proc,
                  'n_speakers': args.n_speakers,
                  'num_train_ep': args.num_train_ep,
                  'num_pretrain_ep': args.num_pretrain_ep,
                  'batch_size_train': args.batch_size,
                  'batch_size_pretrain': args.batch_size_pretrain,
                  'bottleneck_dim': args.bottleneck_dim}
        wandb.init(config=config)
        config['run_id'] = wandb.run.id
        config['last_epoch'] = 0
        config['best'] = best


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


    gpu_queue = Queue(500)
    gpu_proc = Process(target=gpu_logger, args=(gpu_queue,))
    gpu_proc.start()

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

    # ==> load pre-trained model 
    initial_epoch = True
    if args.resume:
        weights_file = wandb.restore('weights.h5')
        network.load_weights(weights_file.name)
        network.save_weights('weights.h5')
        initial_epoch = False

    print(network.summary())
    print('==> gpu {} is, training using loss: {}, aggregation: {}'.format(args.gpu, args.loss, args.aggregation_mode))

    model_path, _ = set_path(args)
    
    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    save_best = keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{acc:.3f}.h5'),
                                                monitor='loss', mode='min', save_best_only=True)

    callbacks = [save_best, normal_lr]

    initial_weights = network.layers[-2].layers[-1].get_weights()

    weight_values = K.batch_get_value(getattr(network.optimizer, 'weights'))

    for epoch in range(last_epoch, args.epochs):
        start_time = time.time()
        pre_t = 0
        pre_h = None

        pre_gpu = [0,0,0,0]

        if not initial_epoch:
            
            network, network_eval = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1),
                                                                 num_class=args.n_speakers,
                                                                 mode='train', args=args)

            network.load_weights('weights.h5')

            # make all layers except the last and first (input layer) one untrainable
            for layer in network.layers[-2].layers[1:-1]:
                layer.trainable = False
            network.layers[-2].layers[-1].set_weights(initial_weights)

            network.compile(optimizer=keras.optimizers.Adam(lr=step_decay(epoch*2)), 
                            loss='categorical_crossentropy', 
                            metrics=['acc'])


            print("==> starting pretrain phase")
            _ = clear_queue(gpu_queue)
            s = time.time()
            pre_h = network.fit_generator(trn_gen,
                                          steps_per_epoch=trn_gen.steps_per_epoch,
                                          epochs=args.num_pretrain_ep,
                                          verbose=1).history
            pre_t = time.time() - s
            pre_gpu = clear_queue(gpu_queue)


            trn_gen.set_batch_size(args.batch_size)
            for layer in network.layers[-2].layers[1:-1]:
                layer.trainable = True
            network.compile(optimizer=keras.optimizers.Adam(lr=step_decay(epoch * args.num_train_ep)), 
                            loss='categorical_crossentropy', 
                            metrics=['acc'])
            network._make_train_function()
            network.optimizer.set_weights(weight_values)
        

        print("==> starting training phase")
        _ = clear_queue(gpu_queue)
        s = time.time()
        trn_h = network.fit_generator(trn_gen,
                                      steps_per_epoch=trn_gen.steps_per_epoch,
                                      epochs=(epoch+1) * args.num_train_ep,
                                      initial_epoch=epoch * args.num_train_ep,
                                      callbacks=callbacks,
                                      verbose=1).history
        trn_t = time.time() - s
        trn_gpu = clear_queue(gpu_queue)
        
        lr = step_decay(epoch * args.num_train_ep)

        trn_gen.redraw_speakers(args.batch_size_pretrain)
        
        embeddings = generate_embeddings(network_eval, eval_cb.test_generator)
        eer = calculate_eer(eval_cb.full_list, embeddings)

        if initial_epoch:
            wandb.run.summary['graph'] = wandb.Graph.from_keras(network.layers[-2])

        weight_values = K.batch_get_value(getattr(network.optimizer, 'weights'))
        network.save_weights('weights.h5')

        K.clear_session()
        gc.collect()
        del network
        del network_eval

        hyp_t = time.time() - start_time

        config['last_epoch'] = epoch
        config['best'] = best
        wandb.save('weights.h5')
        best = save_log(eer, lr, best, initial_epoch,
                        trn_h, pre_h, 
                        trn_gpu, pre_gpu, 
                        trn_t, pre_t, hyp_t)
        initial_epoch = False
        with open('previous_run.json', 'w') as fp:
            json.dump(config, fp)



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
