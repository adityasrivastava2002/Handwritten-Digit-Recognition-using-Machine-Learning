import streamlit as st
from keras.models import load_model
from PIL import Image,ImageOps
from keras.utils import img_to_array
model = load_model('mnist.h5')
upload = st.file_uploader("please upload an image")
button = st.button("enter")
if button:
    img = Image.open(upload)
    st.image(img)
    img = img.resize((28,28))
    img = ImageOps.grayscale(img)
    img = img_to_array(img)
    print(img.shape)
    img = img.reshape(1,28,28,1)
    print(img.shape)
    pred = model.predict(img, batch_size=1)
    print(pred.argmax())
    st.text(f'this is a {pred.argmax()}')

