from keras.callbacks import Callback
from test_generator import TestDataGenerator

import wandb
import tqdm
import numpy as np

import sys
sys.path.append('../tool')
import toolkits

def generate_embeddings(model_eval, test_generator):
    test_generator.start()
    embeddings = {}
    for _ in tqdm.tqdm(test_generator.unique_list, ncols=150, ascii=True, desc='==> generate embeddings'):
        audio, sample = test_generator.sample_queue.get()
        embeddings[audio] = model_eval.predict(sample)
    test_generator.terminate()
    return embeddings

def calculate_eer(full_list, embeddings):
    scores, labels = [], []
    for (lb, p1, p2) in full_list:
        v1 = embeddings[p1][0]
        v2 = embeddings[p2][0]
        scores += [np.sum(v1*v2)]
        labels += [lb]
    scores = np.array(scores)
    labels = np.array(labels)
    eer, _ = toolkits.calculate_eer(labels, scores)
    return eer

def load_verify_list(filename):
    verify_list = np.loadtxt(filename, str)
    verify_lb = [int(i[0]) for i in verify_list]
    list1 = [i[1] for i in verify_list]
    list2 = [i[2] for i in verify_list]
    return list(zip(verify_lb, list1, list2))

def create_unique_list(verify_lists):
    unique_list = []
    for verify_list in verify_lists:
        l1, l2 = list(zip(*verify_list))[1:]
        unique_list.extend(l1)
        unique_list.extend(l2)
    return list(set(unique_list))


class EvalCallback(Callback):
    def __init__(self, model_eval, n_proc, qsize, normalize):
        self.model_eval = model_eval
        self.full_list = load_verify_list('../meta/voxceleb1_veri_test.txt')
        unique_list = create_unique_list([self.full_list])
        self.test_generator = TestDataGenerator(qsize, n_proc, unique_list, normalize)

    def on_epoch_end(self, epoch, logs):
        embeddings = generate_embeddings(self.model_eval, self.test_generator)
        eer = calculate_eer(self.full_list, embeddings)
        wandb.log({'EER': eer}, step=epoch)
        print()
        print('==> Achieved EER: %f'%eer)
        print()