# System
import keras
import numpy as np
import utils as ut
import time
import h5py
import tqdm
import random

from multiprocessing import Process, Queue, Value

def clear_queue(queue):
    try:
        while True:
            queue.get(timeout=5)
    except:
        pass

def enqueue_samples(index_queue, sample_queue, terminator, h5_path, normalize):
    with h5py.File(h5_path, 'r') as data:
        while not terminator.value == 1:
            try:
                speaker, audio_name, idx, length = index_queue.get(timeout=4)
                    
                sample = data['data/' + speaker][idx][:].reshape((257, length))
                sample = np.append(sample, sample[:,::-1], axis=1)
                if normalize:
                    mu = np.mean(sample, 0, keepdims=True)
                    std = np.std(sample, 0, keepdims=True)
                    sample = (sample - mu) / (std + 1e-5)
                sample = sample.reshape((1, 257, 2 * length, 1))
                sample_queue.put((speaker+'/'+audio_name, sample))
            except:
                pass

def enqueue_indices(index_queue, index_list):
    for index in index_list:
        index_queue.put(index)

class TestDataGenerator():
    def __init__(self, qsize, n_proc, normalize=True):
        print('==> Setup Testing Data Generator')
        self.h5_path = '/cluster/home/neurudan/datasets/vox1/vox1_vgg.h5'
        self.normalize = normalize

        self.sample_enqueuers = []
        
        self.terminator = Value('i', 0)

        self.index_queue = Queue(300)
        self.sample_queue = Queue(qsize)
        
        self.start(n_proc)

    def build_index_list(self, unique_list):
        clear_queue(self.index_queue)
        clear_queue(self.sample_queue)
        
        self.unique_list = unique_list
        speakers = {}
        for ID in unique_list:
            speaker, audio_name = ID.split('/')[0], '/'.join(ID.split('/')[1:])
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(audio_name)
        self.index_list = []
        with h5py.File(self.h5_path, 'r') as data:
            for speaker in tqdm.tqdm(speakers, ncols=150, ascii=True, desc='==> Gather Sample Information'):
                audio_names = list(data['audio_names/'+speaker])
                for audio_name in speakers[speaker]:
                    idx = audio_names.index(audio_name)
                    length = data['statistics/'+speaker][idx]
                    self.index_list.append((speaker, audio_name, idx, length))
        self.fill_index_queue()
                    
    def fill_index_queue(self, verbose=False):
        enqueuer = Process(target=enqueue_indices, args=(self.index_queue, self.index_list))
        enqueuer.start()

    def terminate(self):
        self.terminator.value = 1
        clear_queue(self.index_queue)
        clear_queue(self.sample_queue)
        one_alive = True
        while one_alive:
            n = np.sum([1 if t.is_alive() else 0 for t in self.sample_enqueuers])
            one_alive = True if n > 0 else False
            time.sleep(0.01)
        for t in self.sample_enqueuers: t.terminate()
        self.sample_enqueuers = []
        print('==> Testing Enqueuers Terminated')
    
    def start(self, n_proc):
        self.terminator.value = 0
        for _ in range(n_proc):
            args = (self.index_queue, self.sample_queue, self.terminator, self.h5_path, self.normalize)
            enqueuer = Process(target=enqueue_samples, args=args)
            enqueuer.start()
            self.sample_enqueuers.append(enqueuer)