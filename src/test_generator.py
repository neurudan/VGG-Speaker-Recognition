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
    def __init__(self, qsize, n_proc, unique_list, normalize=True):
        print('==> Setup Testing Data Generator')
        self.normalize = normalize
        self.qsize = qsize
        self.n_proc = n_proc
        self.unique_list = unique_list

        self.terminate_enqueuer = False
        self.h5_path = '/cluster/home/neurudan/datasets/vox1/vox1_vgg.h5'

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
        
        self.enqueuers = []

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
                    
    
    def terminate(self):
        print('==> Terminating Testing Enqueuers...')
        self.terminate_enqueuer = True
        one_alive = True
        while one_alive:
            alives = 0
            one_alive = False
            for thread in self.enqueuers:
                if thread.is_alive():
                    alives += 1
                    one_alive = True
            print('%d/%d'%(alives, len(self.enqueuers)))
        for thread in self.enqueuers:
            thread.terminate()
        self.enqueuers = []
        print('==> Testing Enqueuers Terminated')
    
    def start(self):
        self.index_queue = Queue(len(self.index_list))
        for index in self.index_list:
            self.index_queue.put(index)

        self.sample_queue = Queue(self.qsize)
        for _ in range(self.n_proc):
            enqueuer = Process(target=self.enqueue)
            enqueuer.start()
            self.enqueuers.append(enqueuer)