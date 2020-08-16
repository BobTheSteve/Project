import matplotlib.pyplot as plt
import numpy as np

def testingMPL (CIFAR10_files, db_1):
    x = db_1[b"data"]
    x = x.reshape(10000,3,32,32).transpose(0,2,3,1).astype("uint8")
    plt.imshow(x[1])
    plt.show()