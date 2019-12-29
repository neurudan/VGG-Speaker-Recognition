# System
import keras
import numpy as np
import utils as ut
import time
import h5py
import tqdm
import random

from multiprocessing import Process, Queue

class TestDataGenerator():
    def __init__(self, qsize, n_proc, normalize=True):
        print('==> Setup Testing Data Generator')
        self.normalize = normalize
        self.n_proc = n_proc
        self.qsize = qsize

        self.terminate_enqueuer = False
        self.h5_path = '/cluster/home/neurudan/datasets/vox1/vox1_vgg.h5'
        
        self.index_queue = Queue()
        self.enqueuers = []
        self.start()

    def build_index_list(self, unique_list, verbose=False):
        self.unique_list = unique_list
        speakers = {}
        for ID in unique_list:
            speaker, audio_name = ID.split('/')[0], '/'.join(ID.split('/')[1:])
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(audio_name)
        names = []
        self.index_list = []
        with h5py.File(self.h5_path, 'r') as data:
            for speaker in tqdm.tqdm(speakers, ncols=150, ascii=True, desc='==> Gather Sample Information'):
                audio_names = list(data['audio_names/'+speaker])
                for audio_name in speakers[speaker]:
                    idx = audio_names.index(audio_name)
                    length = data['statistics/'+speaker][idx]
                    names.append(speaker+'/'+audio_name)
                    self.index_list.append((speaker, audio_name, idx, length))
        if verbose:
            print('build_index_list')
            print(names)
            print(len(names))
            print(list(set(names)))
            print(len(list(set(names))))

    def enqueue(self):
        with h5py.File(self.h5_path, 'r') as data:
            while not self.terminate_enqueuer:
                try:
                    speaker, audio_name, idx, length = self.index_queue.get(timeout=0.5)
                        
                    sample = data['data/' + speaker][idx][:].reshape((257, length))
                    sample = np.append(sample, sample[:,::-1], axis=1)
                    if self.normalize:
                        mu = np.mean(sample, 0, keepdims=True)
                        std = np.std(sample, 0, keepdims=True)
                        sample = (sample - mu) / (std + 1e-5)
                    sample = sample.reshape((1, 257, 2 * length, 1))

                    self.sample_queue.put((speaker+'/'+audio_name, sample))
                except:
                    pass
                    
    def fill_index_queue(self, verbose=False):
        names = []
        for index in self.index_list:
            names.append(index[0]+'/'+index[1])
            self.index_queue.put(index)
        if verbose:
            print('index_queue')
            print(names)
            print(len(names))
            print(list(set(names)))
            print(len(list(set(names))))

    def terminate(self):
        self.terminate_enqueuer = True
        one_alive = True
        while one_alive:
            one_alive = False
            for thread in self.enqueuers:
                if thread.is_alive():
                    one_alive = True
                    thread.terminate()
        self.enqueuers = []
        print('==> Testing Enqueuers Terminated')
    
    def start(self):
        self.sample_queue = Queue(self.qsize)
        for _ in range(self.n_proc):
            enqueuer = Process(target=self.enqueue)
            enqueuer.start()
            self.enqueuers.append(enqueuer)