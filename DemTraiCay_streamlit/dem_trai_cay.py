import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import io
import os

def main():
    # # Set page configuration
    # st.set_page_config(
    #     page_title="YOLO Object Detector",
    #     page_icon="🔍",
    #     layout="wide"
    # )

    # Title and description
    st.title("YOLO Object Detector")
    st.markdown("Upload an image to detect objects using YOLO model.")

    # Initialize session state for model loading
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None

    # Load model
    @st.cache_resource
    def load_model():
        try:
            model_path = os.path.join(os.path.dirname(__file__), "trai_cay.onnx")
            model = YOLO(model_path, task='detect')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    # Load model if not already loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading YOLO model..."):
            st.session_state.model = load_model()
            if st.session_state.model is not None:
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model. Please check if the model file exists.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])

    # Function to process image
    def process_image(image, confidence=0.5):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_out = image.copy()
        names = st.session_state.model.names
        annotator = Annotator(img_out)

        results = st.session_state.model.predict(image, conf=confidence)

        counts = {}  # Dictionary để đếm số lượng trái cây

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.tolist()

            for box, cls, conf in zip(boxes, clss, confs):
                label = names[int(cls)]
                counts[label] = counts.get(label, 0) + 1  # Đếm số lượng
                annotator.box_label(box, label=f"{label} {conf:.2f}", txt_color=(255, 0, 0), color=(255, 255, 255))

        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        return img_out, counts

    # Confidence slider
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Process uploaded image
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        if st.button("Detect Objects"):
            if st.session_state.model_loaded:
                with st.spinner("Processing image..."):
                    result_image, counts = process_image(image, confidence)

                    st.subheader("Detected Objects")
                    st.image(result_image, use_column_width=True)

                    st.subheader("Fruit Count")
                    if counts:
                        for fruit, count in counts.items():
                            st.write(f"**{fruit}**: {count}")
                    else:
                        st.write("No objects detected.")

                    result_image_pil = Image.fromarray(result_image)
                    buf = io.BytesIO()
                    result_image_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="Download Result",
                        data=byte_im,
                        file_name="detected_objects.png",
                        mime="image/png"
                    )
            else:
                st.error("Model not loaded. Please check if the model file exists.")

    # Add information about the model
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This application uses YOLO (You Only Look Once) for object detection.
        
        Upload an image and click 'Detect Objects' to identify objects in the image.
        
        You can adjust the confidence threshold to filter out low-confidence detections.
        """
    )

    # Add model information
    if st.session_state.model_loaded:
        st.sidebar.header("Model Information")
        st.sidebar.write(f"Model: trai_cay.onnx")
        st.sidebar.write(f"Task: Detection")
        st.sidebar.write(f"Classes: {len(st.session_state.model.names)}") 