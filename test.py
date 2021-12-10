import os 
import numpy as np
import tensorflow as tf 
import cv2 
from data import load_test_data, tf_dataset
from tqdm import tqdm
from tensorflow.keras.models import load_model

from config import *


if __name__ == "__main__":
    ''' Seeding '''
    np.random.seed(42)
    tf.random.set_seed(42)


    ''' Dataset '''
    x_test, y_test = load_test_data(path)

    ''' loading the Model '''
    model = load_model('models/model.h5')


    ''' saving the masks '''
    for x, y in tqdm(zip(x_test, y_test), total=len(x_test)):
        image_name = x.split('/')[-1]
        
        ## Read image
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)


        ## Read mask
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (W, H))   # (256, 256)
        y = y - 1
        y = np.expand_dims(y, -1)  # (256, 256, 1)
        y = y * (255/ num_classes)
        y = y.astype(np.int32)
        y = np.concatenate([y, y, y], axis=2)


        # prediction
        p = model.predict(np.expand_dims(x, axis=0))[0]
        p = np.argmax(p, axis=-1)
        p = np.expand_dims(p, axis=-1)
        p = p * (255/num_classes)
        p = p.astype(np.int32)
        p = np.concatenate([p, p, p], axis=2)

        x = x * 255.0
        x = x.astype(np.int32)

        h, w, _ = x.shape
        line = np.ones((h, 8, 3)) * 255  # white space between the image, mask and prediction

        combined_image = np.concatenate([x, line, y, line, p], axis=1)
        cv2.imwrite(f'results/{image_name}', combined_image)
        


