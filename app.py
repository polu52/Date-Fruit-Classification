import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

model=load_model('date_fruit_model6.h5')

def process_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (170, 170))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Date Fruit Classification ")
st.write('Upload an image and model predict which species of date fruit is this')

file=st.file_uploader('Upload an image',type=['jpg','jpeg','png'])


if file is not None:
    img=Image.open(file)
    st.image(img,caption='Uploaded image')
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)

    class_names={0:'Ajwa',
                 1:'Galaxy',
                 2:'Medjool',
                 3:'Meneifi',
                 4:'Nabtat Ali',
                 5:'Rutab',
                 6:'Shaishe',
                 7:'Sokari',
                 8:'Sugaey'                           
                 }

    st.write(class_names[predicted_class])
