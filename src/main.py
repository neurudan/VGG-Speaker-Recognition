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
import copy
from wandb.keras import WandbCallback

from generator import DataGenerator
from test_generator import TestDataGenerator
from eval_callback import *
from multiprocessing import Process, Queue

sys.path.append('../tool')
import toolkits
import pickle


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
    gpu_log = {'GPU 1': {}, 'GPU 2': {}}
    gpu_log['GPU 1']['Memory'] = np.mean(list(m1))
    gpu_log['GPU 2']['Memory'] = np.mean(list(m2))
    gpu_log['GPU 1']['Usage'] = np.mean(list(u1))
    gpu_log['GPU 2']['Usage'] = np.mean(list(u2))
    return gpu_log

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
parser.add_argument('--resume', action='store_true')
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

def save_tops(val, log, tops, keys, log_tops=True):
    log_name = ' - '.join(keys)
    log[log_name] = val
    if log_tops:
        t = tops
        for k in keys:
            if k not in t:
                t[k] = {}
            t = t[k]
        if 'min' not in t:
            t['min'] = val
            t['max'] = val    
        else:
            t['min'] = np.minimum(val, t['min'])
            t['max'] = np.maximum(val, t['max'])
        log[log_name + ' (min)'] = t['min']
        log[log_name + ' (max)'] = t['max']
    return log, tops

def save_log(tops, initial, 
             lr, eer, 
             h_t, h_p, 
             g_t, g_p, g_e, 
             t_t, t_p, t_e, t_h):
    t_s = t_h - t_t - t_p - t_e
    t_t, t_p, t_e, t_s, t_h = t_t / 60.0, t_p / 60.0, t_e / 60.0, t_s / 60.0, t_h / 60.0
    log = {}
    log, tops = save_tops(lr, log, tops, ['Learn Rate'], log_tops=False)
    log, tops = save_tops(eer, log, tops, ['EER'])
    log, tops = save_tops(t_t, log, tops, ['Train', 'Time needed'])
    for k in [['acc', 'Accuracy'], ['loss', 'Loss']]:
        for i in range(len(h_t[k[0]][:-1])):
            log, tops = save_tops(h_t[k[0]][i], log, tops, ['Train', k[1], f'{i+1}. epoch'])
        log, tops = save_tops(h_t[k[0]][-1], log, tops, ['Train', k[1], 'final epoch'])
        log, tops = save_tops(np.mean(h_t[k[0]]), log, tops, ['Train', k[1], 'mean'])
    
    for k in g_t:
        for m in g_t[k]:
            log, tops = save_tops(g_t[k][m], log, tops, ['Train', k, m])
            log, tops = save_tops(g_e[k][m], log, tops, ['Embeddings', k, m])

    if not initial:
        log, tops = save_tops(t_h, log, tops, ['Hyperepoch', 'Time needed'])
        log, tops = save_tops(t_s, log, tops, ['Setup', 'Time needed'])

        log, tops = save_tops(t_p, log, tops, ['Pretrain', 'Time needed'])
        for k in [['acc', 'Accuracy'], ['loss', 'Loss']]:
            for i in range(len(h_p[k[0]][:-1])):
                log, tops = save_tops(h_p[k[0]][i], log, tops, ['Pretrain', k[1], f'{i+1}. epoch'])
            log, tops = save_tops(h_p[k[0]][-1], log, tops, ['Pretrain', k[1], 'final epoch'])
            log, tops = save_tops(np.mean(h_p[k[0]]), log, tops, ['Pretrain', k[1], 'mean'])
        for k in g_p:
            for m in g_p[k]:
                log, tops = save_tops(g_p[k][m], log, tops, ['Pretrain', k, m])
    wandb.log(log)
    return tops


def main():
    tops = {}
    best_eer = {}
    config = {}
    if args.resume:
        with open('previous_run.pkl', 'rb') as fp:
            data = pickle.load(fp)
            run_id = data['run_id']
            last_epoch = data['last_epoch']
            tops = data['tops']
            best_eer = data['best_eer']

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
        config['tops'] = tops
        config['best_eer'] = best_eer
        last_epoch = 0


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
    
    save_best = keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{acc:.3f}.h5'),
                                                monitor='loss', mode='min', save_best_only=True)

    callbacks = [save_best]

    initial_weights = network.layers[-2].layers[-1].get_weights()



    for epoch in range(last_epoch, args.epochs):
        start_time = time.time()
        pre_t = 0
        pre_h = None
        lr, multiplier = step_decay(epoch)

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

            network.compile(optimizer=keras.optimizers.Adam(lr=lr), 
                            loss='categorical_crossentropy', 
                            metrics=['acc'])

            print("==> starting pretrain phase")
            _ = clear_queue(gpu_queue)
            s = time.time()
            pre_h = network.fit_generator(trn_gen,
                                          steps_per_epoch=trn_gen.steps_per_epoch,
                                          epochs=args.num_pretrain_ep * multiplier,
                                          verbose=1).history
                                          
            pre_t = time.time() - s
            pre_gpu = clear_queue(gpu_queue)
            trn_gen.set_batch_size(args.batch_size)
            
            
            network.save_weights('weights.h5')
            K.clear_session()
            gc.collect()
            del network
            del network_eval

            network, network_eval = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1),
                                                                 num_class=args.n_speakers,
                                                                 mode='train', args=args)

            network.load_weights('weights.h5')

        network.compile(optimizer=keras.optimizers.Adam(lr=lr), 
                        loss='categorical_crossentropy', 
                        metrics=['acc'])
        

        print("==> starting training phase")
        _ = clear_queue(gpu_queue)
        s = time.time()
        trn_h = network.fit_generator(trn_gen,
                                      steps_per_epoch=trn_gen.steps_per_epoch,
                                      epochs=epoch + (args.num_train_ep * multiplier),
                                      initial_epoch=epoch,
                                      callbacks=callbacks,
                                      verbose=1).history
        trn_t = time.time() - s
        trn_gpu = clear_queue(gpu_queue)
        trn_gen.steps_per_epoch

        trn_gen.redraw_speakers(args.batch_size_pretrain)
        

        _ = clear_queue(gpu_queue)
        s = time.time()
        embeddings = generate_embeddings(network_eval, eval_cb.test_generator)
        emb_t = time.time() - s
        emb_gpu = clear_queue(gpu_queue)
        eer = calculate_eer(eval_cb.full_list, embeddings)

        if initial_epoch:
            wandb.run.summary['graph'] = wandb.Graph.from_keras(network.layers[-2])
        
        if lr not in best_eer:
            best_eer[lr] = 0.5
        if best_eer[lr] > eer:
            best_eer[lr] = eer
            network.save_weights(f'best_weights_{lr}.h5')
            wandb.save(f'best_weights_{lr}.h5')
        
        network.save_weights('weights.h5')
        wandb.save('weights.h5')


        K.clear_session()
        gc.collect()
        del network
        del network_eval

        hyp_t = time.time() - start_time

        config['last_epoch'] = epoch
        config['best_eer'] = best_eer

        tops = save_log(tops, initial_epoch,
                        lr, eer,
                        trn_h, pre_h, 
                        trn_gpu, pre_gpu, emb_gpu,
                        trn_t, pre_t, emb_t, hyp_t)

                        
        config['tops'] = tops
        initial_epoch = False
        with open('previous_run.pkl', 'wb') as fp:
            pickle.dump(config, fp)
        wandb.save('previous_run.pkl')



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

    milestone = [stage1, stage2, stage3]
    multipliers = [1, 4, 10] 
    gamma = [1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    multiplier = multipliers[0]
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            multiplier = multipliers[s]
            break
    #print('==> Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr), multiplier


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
