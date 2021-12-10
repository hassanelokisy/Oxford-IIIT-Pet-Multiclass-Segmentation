import streamlit as st 
import numpy as np 
import cv2
from tensorflow.keras.models import load_model
from config import *
from tensorflow.keras.preprocessing.image import array_to_img

st.write("""# Oxford-IIIT Pet Multiclass-Segmentation""")
model = load_model('models/model.h5')


uploaded_file = st.file_uploader("Choose a image", type="jpg")

print(type(uploaded_file))
print(uploaded_file)


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    print("+"*40)

    
    x = cv2.resize(opencv_image, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)

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

    combined_image = np.concatenate([x, line, p], axis=1)

    st.image(array_to_img(combined_image), width=800)