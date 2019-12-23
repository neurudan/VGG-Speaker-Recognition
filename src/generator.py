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
    def __init__(self, dim, mp_pooler, augmentation=True, batch_size=32, nfft=512, spec_len=250,
                 win_length=400, sampling_rate=16000, hop_length=160, n_classes=5994, shuffle=True, normalize=True):
        'Initialization'
        self.spec_len = spec_len
        self.normalize = normalize

        self.shuffle = shuffle
        self.batch_size = batch_size

        self.terminate_enqueuer = False
        self.h5_path = '/cluster/home/neurudan/datasets/vox2/vox2_vgg.h5'

        # Read Speaker List
        lines = []
        with open('../meta/vox2_speakers_5994_dev.txt') as f:
            lines = f.readlines()
        lines = list(set(lines))
        if '\n' in lines:
            lines.remove('\n')
        self.speakers = []
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            self.speakers.append(line)

        self.n_classes = len(self.speakers)

        self.list_IDs = []
        with h5py.File(self.h5_path, 'r') as data:
            for speaker in tqdm(self.speakers, ncols=100, ascii=True, desc='build speaker statistics'):
                times = data['statistics/'+speaker][:, 0]
                speakers = [speaker] * len(times)
                ids = np.arange(len(times))
                self.list_IDs.extend(list(zip(speakers, ids, times)))

        self.index_queue = Queue(self.__len__())
        self.on_epoch_end()

        self.enqueuers = []
        self.sample_queue = Queue(100)
        self.start_enqueuers()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def enqueue(self):
        with h5py.File(self.h5_path, 'r') as data:
            while not self.terminate_enqueuer:
                samples = []
                labels = []
                keys = self.index_queue.get()
                for speaker, idx, length in keys:
                    labels.append(self.speakers.index(speaker))
                    
                    sample = None
                    start = np.random.randint(length*2 - self.spec_len)
                    if start >= length:
                        start = start - length
                        sample = data['data/' + speaker][idx][:, start:start+self.spec_len]
                    elif start + self.spec_len < length:
                        sample = data['data/' + speaker][idx][:, start:start+self.spec_len]
                    else:
                        sample1 = data['data/' + speaker][idx][:, start:]
                        sample2 = data['data/' + speaker][idx][:, :start-length+self.spec_len]
                        sample = np.append(sample1, sample2, axis=1)
                    
                    if np.random.random() < 0.3:
                        sample = sample[:,::-1]                    

                    mu = np.mean(sample, 0, keepdims=True)
                    std = np.std(sample, 0, keepdims=True)

                    samples.append((sample - mu) / (std + 1e-5))
                labels = np.eye(self.n_classes)[labels]
                self.sample_queue.put((np.array(samples), labels))

    def __getitem__(self, index):
        X, y = self.sample_queue.get()
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        random.shuffle(self.list_IDs)
        for i in range(self.__len__()):
            self.index_queue.put(self.list_IDs[i*self.batch_size:(i*self.batch_size)+self.batch_size])

    def start_enqueuers(self):
        for _ in range(16):
            enqueuer = Process(target=self.enqueue)
            enqueuer.start()
            self.enqueuers.append(enqueuer)


def OHEM_generator(model, datagen, steps, propose_time, batch_size, dims, nclass):
    # propose_time : number of candidate batches.
    # prop : the number of hard batches for training.
    step = 0
    interval = np.array([i*(batch_size // propose_time) for i in range(propose_time)] + [batch_size])

    while True:
        if step == 0 or step > steps - propose_time:
            step = 0
            datagen.on_epoch_end()

        # propose samples,
        samples = np.empty((batch_size,) + dims)
        targets = np.empty((batch_size, nclass))

        for i in range(propose_time):
            x_data, y_data = datagen.__getitem__(index=step+i)
            preds = model.predict(x_data, batch_size=batch_size)   # prediction score
            errs = np.sum(y_data * preds, -1)
            err_sort = np.argsort(errs)

            indices = err_sort[:(interval[i+1]-interval[i])]
            samples[interval[i]:interval[i+1]] = x_data[indices]
            targets[interval[i]:interval[i+1]] = y_data[indices]

        step += propose_time
        yield samples, targets
