import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9

def main():
    st.title('Xử lý ảnh số')
    st.sidebar.title('Menu')

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "tif", "bmp", "png", "jpeg", "webp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(imgin, caption='ImageIn', use_column_width=True)

        tab1, tab2, tab3, tab4 = st.tabs(["Chapter3", "Chapter4", "Chapter9", "Yolo"])

        with tab1:
            st.header("Chapter3")
            if st.button("Negative"):
                imgout = c3.Negative(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Negative Color"):
                imgout = c3.NegativeColor(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Logarit"):
                imgout = c3.Logarit(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Power"):
                imgout = c3.PowerLaw(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Piecewise Line"):
                imgout = c3.PiecewiseLine(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Histogram"):
                imgout = c3.Histogram(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Histogram Equal"):
                imgout = c3.HistogramEqualization(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Histogram Equal Color"):
                imgout = c3.HistogramEqualizationColor(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Local Histogram"):
                imgout = c3.LocalHistogram(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Histogram Stat"):
                imgout = c3.HistStat(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Smooth Box"):
                imgout = c3.SmoothBox(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Median Filter"):
                imgout = cv2.medianBlur(imgin, 3)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Sharp"):
                imgout = c3.Sharp(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Gradient"):
                imgout = c3.Gradient(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

        with tab2:
            st.header("Chapter4")
            if st.button("Spectrum"):
                imgout = c4.Spectrum(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Draw Notch Filter"):
                imgout = c4.DrawNotchFilter(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Remove Notch Simple"):
                imgout = c4.RemoveNotchSimple(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Draw Notch Period Filter"):
                imgout = c4.DrawNotchPeriodFilter(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Remove Period Noise"):
                imgout = c4.RemovePeriodNoise(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

        with tab3:
            st.header("Chapter9")
            if st.button("Erosion"):
                imgout = c9.Erosion(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Dilation"):
                imgout = c9.Dilation(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Boundary"):
                imgout = c9.Boundary(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Contour"):
                imgout = c9.Contour(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Convex Hull"):
                imgout = c9.ConvexHull(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Defect Detect"):
                imgout = c9.DefectDetect(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Hole Fill"):
                imgout = c9.HoleFill(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Connected Components"):
                imgout = c9.ConnectedComponents(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

            if st.button("Remove Small Rice"):
                imgout = c9.RemoveSmallRice(imgin)
                st.image(imgout, caption='ImageOut', use_column_width=True)

        with tab4:
            st.header("Yolo")
            if st.button("Predict"):
                names = model.names
                imgout = imgin.copy()
                annotator = Annotator(imgout)
                results  = model.predict(imgin, conf = 0.5, verbose=False)

        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.tolist()
        for box, cls, conf in zip(boxes, clss, confs):
            annotator.box_label(box,label = names[int(cls)] + ' %4.2f' % conf , txt_color=(255,0,0), color=(255,255,255))
                st.image(imgout, caption='ImageOut', use_column_width=True)

if __name__ == "__main__":
    main()
