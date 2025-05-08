import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def main():
    st.title("Object Detection using YOLOv8")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = YOLO('yolov8n.pt')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif", "webp"])

    # Create two columns for image display
    col_img1, col_img2 = st.columns(2)

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        with col_img1:
            st.subheader("Original Image")
            st.image(image, channels="BGR", width=400)

        # Confidence threshold slider
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

        # Process image when button is clicked
        if st.button("Detect Objects"):
            # Make prediction
            results = st.session_state.model.predict(image, conf=conf_threshold, verbose=False)
            
            # Get the first result
            result = results[0]
            
            # Create annotated image
            annotated_img = image.copy()
            annotator = Annotator(annotated_img)
            
            # Get boxes, classes and confidence scores
            boxes = result.boxes.xyxy.cpu()
            clss = result.boxes.cls.cpu().tolist()
            confs = result.boxes.conf.tolist()
            
            # Draw boxes and labels
            for box, cls, conf in zip(boxes, clss, confs):
                annotator.box_label(box, 
                                  label=f"{st.session_state.model.names[int(cls)]} {conf:.2f}",
                                  txt_color=(255, 0, 0),
                                  color=(255, 255, 255))
            
            # Display processed image
            with col_img2:
                st.subheader("Detected Objects")
                st.image(annotated_img, channels="BGR", width=400)
                
                # Display detection information
                st.write("Detection Results:")
                for cls, conf in zip(clss, confs):
                    st.write(f"- {st.session_state.model.names[int(cls)]}: {conf:.2f}")

                # Download button for processed image
                if st.button("Download Processed Image"):
                    is_success, buffer = cv2.imencode(".png", annotated_img)
                    if is_success:
                        st.download_button(
                            label="Download",
                            data=buffer.tobytes(),
                            file_name="detected_objects.png",
                            mime="image/png"
                        )

if __name__ == "__main__":
    main() 