# System
import keras
import numpy as np
import utils as ut
import time
import h5py
import tqdm
import random

from multiprocessing import Process, Queue

class DataGenerator():
    def __init__(self, unique_list, normalize=True):
        self.normalize = normalize

        self.terminate_enqueuer = False
        self.h5_path = '/cluster/home/neurudan/datasets/vox1/vox1_vgg.h5'

        speakers = {}
        for ID in unique_list:
            speaker, audio_name = ID.split('/')[0], '/'.join(ID.split('/')[1:])
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(audio_name)

        self.index_queue = Queue(len(unique_list))
        with h5py.File(self.h5_path, 'r') as data:
            for speaker in tqdm.tqdm(speakers, ncols=100, ascii=True, desc='build speaker statistics'):
                audio_names = list(data['audio_names/'+speaker])
                for audio_name in speakers[speaker]:
                    idx = audio_names.index(audio_name)
                    length = data['statistics/'+speaker][idx]
                    self.index_queue.put((speaker, audio_name, idx, length))
        
        self.enqueuers = []
        self.sample_queue = Queue(100)
        for _ in range(32):
            enqueuer = Process(target=self.enqueue)
            enqueuer.start()
            self.enqueuers.append(enqueuer)

    def enqueue(self):
        try:
            with h5py.File(self.h5_path, 'r') as data:
                while not self.terminate_enqueuer:
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