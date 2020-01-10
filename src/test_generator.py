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
            queue.get(timeout=10)
    except:
        pass


class TestDataGenerator():
    def __init__(self, qsize, n_proc, normalize=True):
        print('==> Setup Testing Data Generator')
        self.h5_path = '/cluster/home/neurudan/datasets/vox1/vox1_vgg.h5'
        self.normalize = normalize

        self.terminator = Value('i', 0)
        self.sample_queue = Queue(qsize)
        self.index_queue = Queue()
        self.enqueuers = []
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

    def enqueue(self, terminator):
        with h5py.File(self.h5_path, 'r') as data:
            while not terminator.value == 1:
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
        for index in self.index_list:
            self.index_queue.put(index)

    def terminate(self):
        self.terminator.value = 1
        one_alive = True
        while one_alive:
            n = np.sum([1 if t.is_alive() else 0 for t in self.enqueuers])
            one_alive = True if n > 0 else False
            time.sleep(0.01)
        for t in self.enqueuers: t.terminate()
        self.enqueuers = []
        print('==> Testing Enqueuers Terminated')
    
    def start(self, n_proc):
        self.terminator.value = 0
        for _ in range(n_proc):
            enqueuer = Process(target=self.enqueue, args=(self.terminator,))
            enqueuer.start()
            self.enqueuers.append(enqueuer)