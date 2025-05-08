import streamlit as st
import os
import sys

# --- Set up page ---
st.set_page_config(page_title="Xu Ly Anh So", layout="wide")
st.sidebar.title("🧭 22110260 - Nguyen Ha Hong Tuan")

# --- Function list ---
function_list = [
    "🧮 Solve Quadratic Equation",
    "🙂 Face Recognition",
    "🍊 Fruit Detection (YOLOv8)",
    "🔢 Count Objects",
    "✍️ Handwritten Digit Recognition",
    "🖼️ Background Removal (u2netp)",
    "🎨 Paint Transfer"
]

# --- Sidebar Menu ---
choice = st.sidebar.radio("Select a function:", function_list)

# --- Dynamic importing based on choice ---
if choice == "🖼️ Background Removal (u2netp)":
    sys.path.append(os.path.join(os.getcwd(), 'BackgroundRemoval_u2netp_streamlit'))
    from BackgroundRemoval_u2netp_streamlit import app as bg_app
    bg_app.main()

elif choice == "🔢 Count Objects":
    sys.path.append(os.path.join(os.getcwd(), 'CountObject_streamlit'))
    from DemTraiCay_streamlit import dem_trai_cay as count_app
    count_app.main()

elif choice == "🧮 Solve Quadratic Equation":
    sys.path.append(os.path.join(os.getcwd(), 'GiaiPtBac2_streamlit'))
    from GiaiPtBac2_streamlit import giai_pt_bac_2 as quadratic_solver
    quadratic_solver.main()

elif choice == "✍️ Handwritten Digit Recognition":
    sys.path.append(os.path.join(os.getcwd(), 'NhanDangChuVietTay_mnist_streamlit'))
    from NhanDangChuVietTay_mnist_streamlit import app as digit_app
    digit_app.main()

elif choice == "🙂 Face Recognition":
    sys.path.append(os.path.join(os.getcwd(), 'NhanDangKhuonMat_onnx_streamlit'))
    from NhanDangKhuonMat_onnx_streamlit import predict as face_app
    face_app.main()

elif choice == "🍊 Fruit Detection (YOLOv8)":
    sys.path.append(os.path.join(os.getcwd(), 'NhanDangTraiCay_yolov8n_streamlit'))
    from NhanDangTraiCay_yolov8n_streamlit import nhan_dang_trai_cay as fruit_app
    fruit_app.main()

elif choice == "🎨 Paint Transfer":
    sys.path.append(os.path.join(os.getcwd(), 'PaintTransfer_streamlit'))
    from PaintTransfer_streamlit import image_to_art as paint_app
    paint_app.main()