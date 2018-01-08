from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
# python lm.py


fd = open(os.path.join('./MNIST/train-images-idx3-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
print(loaded)
trX = loaded[16:].reshape((60000, 28, 28)).astype(np.float)


print('Train Datasets Shape:', trX.shape)


for i in range(trX.shape[0]):
    scipy.misc.imsave('./MNISTIM/B_{}.jpg'.format(i), trX[i])



if __name__ == '__main__':

	pass

# python lm.py