from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

folders = [
    Path("dataset/laptop_open"),
    Path("dataset/laptop_closed")
]

for folder in folders:
    for file in folder.iterdir():
        if file.suffix.lower() in [".heif", ".heic"]:
            img = Image.open(file).convert("RGB")

            new_file = file.with_suffix(".jpg")
            img.save(new_file, "JPEG", quality=95)

            print(f"Converted: {file.name} -> {new_file.name}")
            