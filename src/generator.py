# System
import keras
import numpy as np
import utils as ut
import time
import h5py
import tqdm

from multiprocessing import Process, Queue

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, dim, mp_pooler, augmentation=True, batch_size=32, nfft=512, spec_len=250,
                 win_length=400, sampling_rate=16000, hop_length=160, n_classes=5994, shuffle=True, normalize=True):
        'Initialization'
        self.spec_len = spec_len
        self.normalize = normalize

        self.labels = labels
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.augmentation = augmentation

        self.on_epoch_end()

        # put dataset path here
        self.h5_path = '/cluster/home/neurudan/datasets/vox2/vox2_vgg.h5'

        speakers = {}
        for i, ID in enumerate(list_IDs):
            speaker = ID.split('/')[0]
            file_name = ID.split('/')[1] + '/' + ID.split('/')[2]
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append((file_name, labels[i]))

        self.sample_allocation = {}
        with h5py.File(self.h5_path, 'r') as data:
            for speaker in tqdm.tqdm(speakers, ncols=100, ascii=True, desc='build speaker statistics'):
                names = names.append(data['audio_names/'+speaker][:,0]
                print(names.shape)
                for audio, speaker_id in speakers[speaker]:
                    idx = names.index(audio)
                    length = data['statistics/'+speaker][idx, 0]
                    self.sample_allocation[speaker+'/'+audio] = (speaker, idx, speaker_id, length)
        
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
                for _ in range(self.batch_size):
                    ID = self.index_queue.get()
                    speaker, idx, speaker_id, length = self.sample_allocation[ID]
                    labels.append(speaker_id)
                    
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
        max_items = self.__len__()*self.batch_size
        self.index_queue = Queue(max_items)
        IDs = self.list_IDs.copy()
        if self.shuffle:
            np.random.shuffle(IDs)
        for i in range(max_items):
            self.index_queue.put(IDs[i])

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
