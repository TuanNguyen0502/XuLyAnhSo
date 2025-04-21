import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import io

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((320, 320))
    img_np = np.array(image).astype(np.float32) / 255.0  # chuẩn hóa về [0,1]
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
    img_np = np.expand_dims(img_np, axis=0)   # (1,3,320,320)
    return img_np

# Hàm hậu xử lý mask
def postprocess_mask(mask, orig_size):
    mask = mask[0, 0]  # lấy channel đầu tiên
    mask = cv2.resize(mask, orig_size)
    mask = (mask * 255).astype(np.uint8)
    return mask

# Load model ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("u2netp.onnx")

# Streamlit UI
st.title("🖼️ Background Removal with U2NETP (ONNX)")
st.write("Upload ảnh vào để tách nền!")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Tiền xử lý
    img_input = preprocess_image(image)

    # Load model
    session = load_model()

    # Dự đoán mask
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img_input})
    mask = result[0]

    # Hậu xử lý mask
    orig_size = image.size
    mask = postprocess_mask(mask, orig_size)

    # Tạo ảnh mới đã tách nền
    image_np = np.array(image)
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result_image = cv2.bitwise_and(image_np, mask_3c)

    st.image(mask, caption="Mask", use_column_width=True)
    st.image(result_image, caption="Ảnh đã tách nền", use_column_width=True)

    # Cho phép tải về
    result_pil = Image.fromarray(result_image)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Tải ảnh đã tách nền",
        data=byte_im,
        file_name="removed_background.png",
        mime="image/png"
    )
