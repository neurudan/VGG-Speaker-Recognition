# System
import keras
import numpy as np
import utils as ut
import time
import h5py
import tqdm
import random

from multiprocessing import Process, Queue

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, n_proc, qsize, batch_size=32, spec_len=250, normalize=True):
        'Initialization'
        
        print('==> Setup Training Data Generator')
        self.spec_len = spec_len
        self.normalize = normalize

        self.batch_size = batch_size
        self.qsize = qsize
        self.n_proc = n_proc

        self.terminate_enqueuer = False
        self.h5_path = '/cluster/home/neurudan/datasets/vox2/vox2_vgg.h5'

        # Read Speaker List
        self.speakers = []
        with open('../meta/vox2_speakers_5994_dev.txt') as f:
            lines = list(set(f.readlines()))
            if '\n' in lines:
                lines.remove('\n')
            for line in lines:
                if line[-1] == '\n':
                    line = line[:-1]
                self.speakers.append(line)

        # Generate Sample Statistics
        self.list_IDs = []
        with h5py.File(self.h5_path, 'r') as data:
            for speaker in tqdm.tqdm(self.speakers, ncols=150, ascii=True, desc='==> Gather Sample Information'):
                times = []
                speakers = []
                ids = []
                for i, time in enumerate(data['statistics/'+speaker][:]):
                    if time * 2 > self.spec_len:
                        times.append(time)
                        ids.append(i)
                        speakers.append(speaker)
                self.list_IDs.extend(list(zip(speakers, ids, times)))

        self.n_classes = len(self.speakers)
        self.steps_per_epoch = int(np.floor(len(self.list_IDs) / self.batch_size))

        self.start()

    def __len__(self):
        return self.steps_per_epoch

    def enqueue(self):
        with h5py.File(self.h5_path, 'r') as data:
            while not self.terminate_enqueuer:
                samples = []
                labels = []
                keys = self.index_queue.get()
                for speaker, idx, length in keys:
                    labels.append(self.speakers.index(speaker))
                    
                    start = np.random.randint(length*2 - self.spec_len)
                    sample = data['data/' + speaker][idx][:].reshape((257, length))
                    sample = np.append(sample, sample, axis=1)[:,start:start+self.spec_len]
                    
                    if np.random.random() < 0.3:
                        sample = sample[:,::-1]

                    if self.normalize:
                        mu = np.mean(sample, 0, keepdims=True)
                        std = np.std(sample, 0, keepdims=True)
                        sample = (sample - mu) / (std + 1e-5)
                    samples.append(sample)
                labels = np.eye(self.n_classes)[labels]
                samples = np.array(samples)
                samples = samples.reshape(samples.shape+(1,))
                self.sample_queue.put((samples, labels))

    def index_enqueuer(self):
        while not self.terminate_enqueuer:
            random.shuffle(self.list_IDs)
            for i in range(self.__len__()):
                self.index_queue.put(self.list_IDs[i*self.batch_size:(i*self.batch_size)+self.batch_size])


    def __getitem__(self, index):
        X, y = self.sample_queue.get()
        return X, y

    def on_epoch_end(self):
        pass

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
        print('==> Training Enqueuers & Indexer Terminated')

    def start(self):
        self.index_queue = Queue(self.__len__())
        indexer = Process(target=self.index_enqueuer)
        indexer.start()
        
        self.enqueuers = [indexer]
        self.sample_queue = Queue(self.qsize)
        for _ in range(self.n_proc):
            enqueuer = Process(target=self.enqueue)
            enqueuer.start()
            self.enqueuers.append(enqueuer)

