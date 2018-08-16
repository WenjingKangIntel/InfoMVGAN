from PIL import Image
import numpy as np
from utils import *
import os


OUTPUT_SIZE = 128
rows = 64
cols = 64
data_path = '/home/sunliang/dataset/img_align_celeba/'
def read_data():

    images_name = []
    images_v1 = []
    images_v2 = []
    f = open(r'/home/sunliang/dataset/list_attr_celeba.txt')
    lines = f.readlines()
    #for i in range(2, 202599 + 2):
    for i in range(2, 300):
        line = lines[i].split()
        images_name.append(line[0])

    for img_name in images_name:
        img_path = data_path + img_name
        img = Image.open(img_path)

        h, w = img.size[:2]
        j, k = (h - OUTPUT_SIZE) / 2, (w - OUTPUT_SIZE) / 2
        box = (j, k, j + OUTPUT_SIZE, k + OUTPUT_SIZE)
        img = img.crop(box=box)
        img = img.resize((rows, cols))

        img_v1 = np.array(img)
        img_v2 = np.array(img)
        for row in range(16, 48):
            for col in range(16, 48):
                img_v2[row][col] = 127
        images_v1.append(img_v1)
        images_v2.append(img_v2)


    print 'load data OK'
    images_v1 = np.array(images_v1)
    images_v2 = np.array(images_v2)
    return images_v1, images_v2


def read_test_data():
    test_data_path = os.getcwd() + "/test_data/"
    test_data = []
    for img_name in os.listdir(test_data_path):
        img_path = test_data_path+img_name
        img = Image.open(img_path)
        img = np.array(img)
        for i in range(16):
            test_data.append(img)

    return np.array(test_data)


if __name__ == '__main__':
    x_v1, x_v2 = read_data()

    sample_images_v1 = (x_v1[0: 64]-127.5)/127.5
    sample_images_v2 = (x_v2[0: 64] - 127.5) / 127.5
    save_images(sample_images_v1, [8, 8], os.getcwd() + '/samples/' + 'images_v1.png')
    save_images(sample_images_v2, [8, 8], os.getcwd() + '/samples/' + 'images_v2.png')
