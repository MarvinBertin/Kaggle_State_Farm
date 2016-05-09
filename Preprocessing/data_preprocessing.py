from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
#from scipy.io import imread, imsave
from scipy.misc import imresize, imsave
from scipy.ndimage import imread

flag_subset = False
flag_ds = 8

WIDTH, HEIGHT = 640 // 8, 480 // 8
NUM_CLASSES = 10

def load_image(path):
    return imresize(imread(path), (HEIGHT, WIDTH))

def load_train(base):
    driver_imgs_list = pd.read_csv('/home/ubuntu/driver_imgs_list.csv')
    driver_imgs_grouped = driver_imgs_list.groupby('classname')

    X_train = []
    y_train = []
    driver_ids = []

    print('Reading train images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        driver_ids_group = driver_imgs_grouped.get_group('c{}'.format(j))
        paths = os.path.join(base, 'c{}/'.format(j)) + driver_ids_group.img

        if flag_subset:
            paths = paths[:100]
            driver_ids_group = driver_ids_group.iloc[:100]

        driver_ids += driver_ids_group['subject'].tolist()

        for i, path in tqdm(enumerate(paths), total=len(paths)):
            img = load_image(path)
            if i == 0:
                imsave('c{}.jpg'.format(j), img)

            X_train.append(img)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = OneHotEncoder(n_values=NUM_CLASSES) \
        .fit_transform(y_train.reshape(-1, 1)) \
        .toarray()

    return X_train, y_train, driver_ids

def load_test(base):
    X_test = []
    X_test_id = []
    paths = glob.glob('{}*.jpg'.format(base))

    if flag_subset:
        paths = paths[:100]

    print('Reading test images...')
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        id = os.path.basename(path)
        img = load_image(path)

        X_test.append(img)
        X_test_id.append(id)

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    return X_test, X_test_id

X_train, y_train, driver_ids = load_train('/home/ubuntu/data/train/')
X_test, X_test_ids = load_test('/home/ubuntu/data/test/')
dest = 'data_{}.pkl'.format(flag_ds)

with open(dest, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, X_test_ids, driver_ids), f)
