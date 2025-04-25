import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import io
import numpy as np
# Thêm dòng này vào đầu file image_to_art.py
from transformer_model import TransformerNet

# ===================== Preprocessing =====================
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def test_transform(image_size=None):
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform

def denormalize(tensor):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

# ===================== Main App =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.title("🎨 Image to Art with Style Transfer")
    st.markdown("Upload an image and convert it to a painting style using a pre-trained model.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    model_path = os.path.join(os.path.dirname(__file__), "Picasso_Selfportrait_5000.pth")

    def stylize_image(image, model_path):
        transformer = TransformerNet().to(device)
        transformer.load_state_dict(torch.load(model_path, map_location=device))
        transformer.eval()

        transform = test_transform()
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = denormalize(transformer(image_tensor)).cpu()

        # Chuyển về định dạng ảnh hiển thị trên Streamlit
        output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
        return output_image

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        if not os.path.exists(model_path):
            st.error("Model file not found. Please check the path.")
            return

        if st.button("Stylize"):
            with st.spinner("Applying style..."):
                stylized_image = stylize_image(image, model_path)

                # Convert NumPy array to PIL Image
                stylized_image_pil = Image.fromarray((stylized_image * 255).astype(np.uint8))

                st.image(stylized_image_pil, caption="Stylized Image", use_column_width=True)

                buf = io.BytesIO()
                # stylized_image.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Stylized Image",
                    data=byte_im,
                    file_name="stylized_output.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()
