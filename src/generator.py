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

class DataGenerator(keras.utils.Sequence):

    def __init__(self, n_proc, qsize, n_speakers, batch_size=32, spec_len=250, normalize=True):
        print('==> Setup Training Data Generator')
        self.h5_path = '/cluster/home/neurudan/datasets/vox2/vox2_vgg.h5'

        self.spec_len = spec_len
        self.normalize = normalize
        self.n_speakers = n_speakers
        self.batch_size = batch_size

        self.all_speakers = []
        self.speaker_statistics = {}

        # Read Speaker List
        with open('../meta/vox2_speakers_5994_dev.txt') as f:
            lines = list(set(f.readlines()))
            if '\n' in lines:
                lines.remove('\n')
            for line in lines:
                if line[-1] == '\n':
                    line = line[:-1]
                self.all_speakers.append(line)

        # Generate Sample Statistics
        with h5py.File(self.h5_path, 'r') as data:
            for speaker in tqdm.tqdm(self.all_speakers, ncols=150, ascii=True, desc='==> Gather Sample Information'):
                idxs, times = [], []
                for i, time in enumerate(data['statistics/'+speaker][:]):
                    if time * 2 > self.spec_len:
                        idxs.append(i)
                        times.append(time)
                self.speaker_statistics[speaker] = (idxs, times)

        self.sample_enqueuers = []

        self.index_enqueuer_terminator = Value('i', 0)
        self.sample_enqueuer_terminator = Value('i', 0)

        self.index_queue = Queue(qsize)
        self.sample_queue = Queue(qsize)

        self.redraw_speakers()
        self.start(n_proc)


    def redraw_speakers(self):
        self.index_enqueuer_terminator.value = 1
        clear_queue(self.index_queue)
        clear_queue(self.sample_queue)

        random.shuffle(self.all_speakers)
        speakers = self.all_speakers[:self.n_speakers]

        indices = []
        for i, speaker in enumerate(speakers):
            (idxs, times) = self.speaker_statistics[speaker]
            labels, speakers = [i] * len(idxs), [speaker] * len(idxs)
            indices.extend(list(zip(labels, speakers, idxs, times)))

        self.steps_per_epoch = int(np.floor(len(indices) / self.batch_size))

        args = (self.index_enqueuer_terminator, self.index_queue, indices, self.batch_size, self.steps_per_epoch)

        self.index_enqueuer_terminator.value = 0
        self.index_enqueuer = Process(target=self.enqueue_indices, args=args)
        self.index_enqueuer.start()

    def start(self, n_proc):
        self.sample_enqueuer_terminator.value = 0
        args = (self.sample_enqueuer_terminator, self.index_queue, self.sample_queue, 
                self.h5_path, self.n_speakers, self.spec_len, self.n_speakers)
        for _ in range(n_proc):
            enqueuer = Process(target=self.enqueue_samples, args=args)
            enqueuer.start()
            self.sample_enqueuers.append(enqueuer)
            
    def terminate(self):
        self.index_enqueuer_terminator.value = 1
        self.sample_enqueuer_terminator.value = 1
        one_alive = True
        while one_alive:
            one_alive = True if np.sum([1 if t.is_alive() else 0 for t in self.sample_enqueuers]) > 0 else False
            time.sleep(0.5)
        self.sample_enqueuers = []

        clear_queue(self.index_queue)
        clear_queue(self.sample_queue)

        print('==> Training Enqueuers & Indexer Terminated')

    def __len__(self):
        return self.steps_per_epoch

    def enqueue_samples(self, terminator, index_queue, sample_queue, h5_path, n_speakers, spec_len, normalize):
        with h5py.File(h5_path, 'r') as data:
            while not terminator.value == 1:
                for _ in range(50):
                    samples = []
                    labels = []
                    for label, speaker, idx, length in index_queue.get():
                        labels.append(label)
                        
                        start = np.random.randint(length*2 - spec_len)
                        sample = data['data/' + speaker][idx][:].reshape((257, length))
                        sample = np.append(sample, sample, axis=1)[:,start:start+spec_len]
                        
                        if np.random.random() < 0.3:
                            sample = sample[:,::-1]

                        if normalize:
                            mu = np.mean(sample, 0, keepdims=True)
                            std = np.std(sample, 0, keepdims=True)
                            sample = (sample - mu) / (std + 1e-5)
                        samples.append(sample)
                    labels = np.eye(n_speakers)[labels]
                    samples = np.array(samples)
                    samples = samples.reshape(samples.shape+(1,))
                    sample_queue.put((samples, labels))

    def enqueue_indices(self, terminator, index_queue, indices, batch_size, steps):
        while not terminator.value == 1:
            for _ in range(50):
                random.shuffle(indices)
                for i in range(steps):
                    if terminator.value == 1:
                        break
                    index_queue.put(indices[i*batch_size:(i*batch_size)+batch_size])

    def __getitem__(self, index):
        return self.sample_queue.get()

    def on_epoch_end(self):
        pass
