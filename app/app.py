from ultralytics import YOLO
import os
import PIL
import streamlit as st

st.title("Furniture Detection App")
st.markdown("An application that enables furniture detection with using deep learning mechanisms. A [YOLOv8](https://docs.ultralytics.com) model is used for object detection.")

confidence = float(st.slider(
    "Select model confidence", 25, 100, 40)) / 100

source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/yolov8-30epochs.pt')

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")

col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is not None:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image")
    except Exception as ex:
        st.error("Error occurred while opening the image.")

if source_img is not None:
    if st.button('Detect', use_container_width=True):
            res = model.predict(uploaded_image, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            with col2:
                st.image(res_plotted, caption='Detected Image')
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.error("No image is uploaded yet!")
