from sklearn.utils import validation
from model import build_unet
from data import load_data, tf_dataset
from config import *

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

if __name__ == '__main__':
    ''' Seeding '''
    np.random.seed(42)
    tf.random.set_seed(42)

    ''' Loading Dataset '''
    (x_train, y_train), (x_validate, y_validate) = load_data(path)
    print(
        f"Dataset: Train: {len(x_train)} - Valid: {len(x_validate)}")

    train_dataset = tf_dataset(x_train, y_train)
    validation_dataset = tf_dataset(x_validate, y_validate)

    train_steps = len(x_train) // batch_size
    validation_steps = len(x_validate) // batch_size

    ''' Model '''
    model = build_unet(shape=shape, num_classes=num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        lr), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    ''' Bulding Callbacks '''
    callbacks = [
        ModelCheckpoint('models/model.h5', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3,
                          factor=0.1, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ]

    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              batch_size=batch_size,
              epochs=epochs, callbacks=callbacks,
              validation_data=validation_dataset,
              validation_steps=validation_steps,
              )
