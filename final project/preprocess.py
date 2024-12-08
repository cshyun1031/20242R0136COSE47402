import os
from PIL import Image
import pillow_heif

DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 32
def convert_heic_to_jpg(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".heic"):
                heic_path = os.path.join(root, file)
                jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
                try:
                    heif_image = pillow_heif.read_heif(heic_path)
                    image = Image.frombytes(
                        heif_image.mode, heif_image.size, heif_image.data
                    )
                    image.save(jpg_path, "JPEG")
                    os.remove(heic_path)  # 원본 파일 삭제
                    print(f"Converted {file} to {jpg_path}")
                except Exception as e:
                    print(f"Failed to convert {file}: {e}")
convert_heic_to_jpg(TRAIN_DIR)
convert_heic_to_jpg(TEST_DIR)

def resize_images_in_folder(folder_path, size=(224, 224)):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    img = img.resize(size)
                    img.save(img_path)
                    print(f"Resized {file} to {size}")
                except Exception as e:
                    print(f"Failed to resize {file}: {e}")

resize_images_in_folder(TRAIN_DIR, size=(224, 224))
resize_images_in_folder(TEST_DIR, size=(224, 224))