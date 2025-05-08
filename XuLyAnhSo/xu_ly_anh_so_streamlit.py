import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9

def main():
    # Initialize session state for images
    if 'imgin' not in st.session_state:
        st.session_state.imgin = None
    if 'imgout' not in st.session_state:
        st.session_state.imgout = None
    if 'file_bytes' not in st.session_state:
        st.session_state.file_bytes = None
    if 'color_imgin' not in st.session_state:
        st.session_state.color_imgin = None

    # Title
    st.title("Xử lý ảnh số")

    # File uploader
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'webp'])

    # Create two columns for image display
    col_img1, col_img2 = st.columns(2)

    if uploaded_file is not None:
        # Store file bytes in session state
        st.session_state.file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Read both grayscale and color versions
        st.session_state.imgin = cv2.imdecode(st.session_state.file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state.color_imgin = cv2.imdecode(st.session_state.file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image in first column
        with col_img1:
            st.subheader("Ảnh gốc")
            # Check if the image is color or grayscale
            if len(st.session_state.color_imgin.shape) == 3:
                st.image(st.session_state.color_imgin, channels="BGR", width=400)
            else:
                st.image(st.session_state.imgin, channels="GRAY", width=400)

    # Create tabs for different processing categories
    tab1, tab2, tab3 = st.tabs(["Chapter 3", "Chapter 4", "Chapter 9"])

    # Chapter 3 Processing
    with tab1:
        st.header("Chapter 3 - Xử lý điểm và không gian")
        if st.session_state.imgin is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Negative"):
                    st.session_state.imgout = c3.Negative(st.session_state.imgin)
                if st.button("Negative Color"):
                    st.session_state.imgout = c3.NegativeColor(st.session_state.color_imgin)
                if st.button("Logarit"):
                    st.session_state.imgout = c3.Logarit(st.session_state.imgin)
                if st.button("Power Law"):
                    st.session_state.imgout = c3.PowerLaw(st.session_state.imgin)
                if st.button("Piecewise Line"):
                    st.session_state.imgout = c3.PiecewiseLine(st.session_state.imgin)
                if st.button("Histogram"):
                    st.session_state.imgout = c3.Histogram(st.session_state.imgin)
                if st.button("Histogram Equalization"):
                    st.session_state.imgout = c3.HistogramEqualization(st.session_state.imgin)
                if st.button("Histogram Equalization Color"):
                    st.session_state.imgout = c3.HistogramEqualizationColor(st.session_state.color_imgin)
                if st.button("Local Histogram"):
                    st.session_state.imgout = c3.LocalHistogram(st.session_state.imgin)
                if st.button("Histogram Stat"):
                    st.session_state.imgout = c3.HistStat(st.session_state.imgin)
                if st.button("Smooth Box"):
                    st.session_state.imgout = c3.SmoothBox(st.session_state.imgin)
                if st.button("Median Filter"):
                    st.session_state.imgout = cv2.medianBlur(st.session_state.imgin, 3)
                if st.button("Sharp"):
                    st.session_state.imgout = c3.Sharp(st.session_state.imgin)
                if st.button("Gradient"):
                    st.session_state.imgout = c3.Gradient(st.session_state.imgin)

    # Chapter 4 Processing
    with tab2:
        st.header("Chapter 4 - Xử lý tần số")
        if st.session_state.imgin is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Spectrum", key="c4_spectrum"):
                    st.session_state.imgout = c4.Spectrum(st.session_state.imgin)
                if st.button("Draw Notch Filter", key="c4_notch"):
                    st.session_state.imgout = c4.DrawNotchFilter(st.session_state.imgin)
                if st.button("Remove Notch Simple", key="c4_remove_notch"):
                    st.session_state.imgout = c4.RemoveNotchSimple(st.session_state.imgin)
                if st.button("Draw Notch Period Filter", key="c4_notch_period"):
                    st.session_state.imgout = c4.DrawNotchPeriodFilter(st.session_state.imgin)
                if st.button("Remove Period Noise", key="c4_remove_period"):
                    st.session_state.imgout = c4.RemovePeriodNoise(st.session_state.imgin)

    # Chapter 9 Processing
    with tab3:
        st.header("Chapter 9 - Xử lý hình thái")
        if st.session_state.imgin is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Erosion", key="c9_erosion"):
                    st.session_state.imgout = c9.Erosion(st.session_state.imgin)
                if st.button("Dilation", key="c9_dilation"):
                    st.session_state.imgout = c9.Dilation(st.session_state.imgin)
                if st.button("Boundary", key="c9_boundary"):
                    st.session_state.imgout = c9.Boundary(st.session_state.imgin)
                if st.button("Contour", key="c9_contour"):
                    st.session_state.imgout = c9.Contour(st.session_state.imgin)
                if st.button("Convex Hull", key="c9_convex"):
                    st.session_state.imgout = c9.ConvexHull(st.session_state.imgin)
                if st.button("Defect Detect", key="c9_defect"):
                    st.session_state.imgout = c9.DefectDetect(st.session_state.imgin)
                if st.button("Hole Fill", key="c9_hole"):
                    st.session_state.imgout = c9.HoleFill(st.session_state.imgin)
                if st.button("Connected Components", key="c9_connected"):
                    st.session_state.imgout = c9.ConnectedComponents(st.session_state.imgin)
                if st.button("Remove Small Rice", key="c9_rice"):
                    st.session_state.imgout = c9.RemoveSmallRice(st.session_state.imgin)

    # Display processed image in second column
    if st.session_state.imgout is not None:
        with col_img2:
            st.subheader("Ảnh đã xử lý")
            # Check if the output image is grayscale or color
            if len(st.session_state.imgout.shape) == 2:
                st.image(st.session_state.imgout, channels="GRAY", width=400)
            else:
                st.image(st.session_state.imgout, channels="BGR", width=400)
            
            # Download button for processed image
            if st.button("Tải xuống ảnh đã xử lý"):
                is_success, buffer = cv2.imencode(".png", st.session_state.imgout)
                if is_success:
                    st.download_button(
                        label="Tải xuống",
                        data=buffer.tobytes(),
                        file_name="processed_image.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main() 