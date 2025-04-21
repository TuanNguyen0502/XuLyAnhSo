from PIL import Image
import numpy as np
import os

# Thư mục chứa ảnh đã tách nền (PNG RGBA)
input_dir = 'dataset/train/masks_input/'
# Thư mục lưu ảnh mask chuẩn
output_dir = 'dataset/masks/'

# Tạo thư mục output nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Duyệt tất cả file trong folder mask
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Đọc ảnh
        img = Image.open(input_path).convert("RGBA")
        alpha = np.array(img)[:, :, 3]  # Lấy kênh alpha

        # Tạo mask: foreground=255, background=0
        mask = (alpha > 0).astype(np.uint8) * 255

        # Lưu ảnh mask chuẩn
        Image.fromarray(mask).save(output_path)

        print(f"✔️ Đã xử lý: {filename}")

print("✅ Chuyển đổi mask hoàn tất!")
