import os 
import numpy as np
from scipy.sparse import data 
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import cv2 
import pandas as pd
from config import *



def process_data(data_path, file_path):
    df = pd.read_csv(file_path, sep=' ', header=None)
    names = df[0].values

    images = [os.path.join(data_path, f'images/{name}.jpg') for name in names]
    masks = [os.path.join(data_path, f'annotations/trimaps/{name}.png') for name in names]

    return images, masks


def load_data(path):
    train_valid_path = os.path.join(path, "annotations/trainval.txt")
    x_train, y_train = process_data(path, train_valid_path)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_validate, y_validate)

def load_test_data(path):
    test_path = os.path.join(path, "annotations/test.txt")
    x_test, y_test = process_data(path, test_path)
    return x_test, y_test 
 
def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x 


def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x - 1
    x = x.astype(np.int32)
    return x 


def tf_dataset(x, y, batch=batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset



def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)
        return image, mask 

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 3])
    
    return image, mask


if __name__ == '__main__':
    path = "oxford-iiit-pet/"
    
    (x_train, y_train), (x_validate, y_validate)= load_data(path)
    print(f"Dataset: Train: {len(x_train)} - Valid: {len(x_validate)} ")

    dataset = tf_dataset(x_train, y_train)
    for x, y in dataset:
        print(x.shape, y.shape)
        break


