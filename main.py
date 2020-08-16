from os import strerror
import time
import tensorflow as tf
from tensorflow import keras
from progress.bar import IncrementalBar

import unpickle
import testingMPL

CIFAR_DIR = '/Users/joeylee/downloads/cifar-10-batches-py/'

CIFAR10_files = ['batches.meta', 'data_batch_1',  'data_batch_2',  'data_batch_3',  'data_batch_4',  'data_batch_5', 'test_batch']
all_data = [0,1,2,3,4,5,6]

for i, direc in zip (all_data, CIFAR10_files):
    all_data[i] = unpickle.unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
db_1 = all_data[1]
db_2 = all_data[2]
db_3 = all_data[3]
db_4 = all_data[4]
db_5 = all_data[5]
tb = all_data[6]
 
testingMPL.testingMPL(CIFAR10_files, db_1)