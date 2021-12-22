import h5py
from tensorflow import keras
import numpy as np
import sys

from matplotlib import image
import collections
import random


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def predict(img):
    if isinstance(img, str):
        img = image.imread(img)
    img = img.reshape(55, 47, 3)

    # compute the original class
    img_y = bd_model.predict(np.array([img]))
    origin = np.argmax(img_y[0])

    # randomly pick one from a certain class
    newxs = []
    for k, v in cls2img.items():
        newxs.append(img + cl_x_test[random.choice(v)])

    newxs = np.array(newxs)
    ps = bd_model.predict(newxs)
    res = [np.argmax(p) for p in ps]

    count = collections.Counter(res)

    entropy = 0
    for k, v in count.items():
        p = v / len(newxs)
        entropy += -p * np.log2(p)

    if entropy > 4:
        return origin
    else:
        return len(newxs)


'''
usage: 
cd lab3
python p_eval.py model_filename data_filename img_filename
output: a number

example: 
cd lab3
python p_eval.py models/bd_net.h5 data/cl/valid.h5 img.png
output: 1283

'''

if __name__ == '__main__':
    clean_data_filename = 'data/cl/valid.h5'
    path_bd = 'models/bd_net.h5'
    image_path = 'res.png'

    path_bd = sys.argv[1]
    clean_data_filename = sys.argv[2]
    image_path = sys.argv[3]

    # init
    bd_model = keras.models.load_model(path_bd)
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    cls2img = collections.defaultdict(list)

    for i, cls in enumerate(cl_y_test):
        cls2img[int(cls)].append(i)

    res = predict(image_path)
    print(res)

    # name = 'data/cl/test.h5'
    # xs, ys = data_loader(name)
    #
    # xs = xs[:200]
    # ys = ys[:200]
    # p = []
    # for i, x in enumerate(xs):
    #     print(f'{i} / {len(xs)}')
    #     p.append(predict(x))
    #
    # print(np.mean(np.equal(p, ys)) * 100)
