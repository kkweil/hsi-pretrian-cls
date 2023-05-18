import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time





if __name__ == '__main__':
    imp = r'../data/Salinas_corrected.mat'
    import torch

    img = loadmat(imp)
    img_name = [t for t in list(img.keys()) if not t.startswith('__')][0]

    img = img[img_name]
    inputs = torch.randn((90, 224, 15, 15))
    start = time.time()
    img_pad = spectral_padding(inputs, 297)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

    # plt.imshow(img_pad[:,:,69])
    plt.plot(inputs.numpy()[50, 50])
    plt.plot(img_pad[50, 50])
    plt.show()

    a = 0
