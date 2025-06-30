# import streamlit as st
# import numpy as np
# from PIL import Image
# import cv2
# from ultralytics import YOLO

# @st.cache_resource
# def load_model():
#     return YOLO(r"C:\Users\ishaan.narayan\Desktop\Ishaan's Workspace\Nlp_DL\DeepLearning-NLP\Yolo-FineTuning-Project\Fracture_yolo_training\exp12\weights\best.pt")
# model = load_model()
# # return model

# st.title("Yolo")
# st.write("Upload Image") 

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='uploaded image',  use_column_width=True)
#     image_np = np.array(image)
#     results = model(image_np)
    
#     results_image = results [0].plot()
#     st.image(results_image, use_column_width=True) 


import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Absolute path to your model
MODEL_PATH = "best.pt"

# Check if the file exists before loading the model
if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found at:\n{MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("Yolo")
st.write("Upload Image") 

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='uploaded image',  use_column_width=True)
    image_np = np.array(image)
    results = model(image_np)
    results_image = results[0].plot()
    st.image(results_image, use_column_width=True)