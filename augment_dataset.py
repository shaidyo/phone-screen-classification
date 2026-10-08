from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

source_root = Path("dataset")
output_root = Path("dataset_augmented")

classes = ["laptop_open", "laptop_closed"]

TARGET_COUNT = 120

for class_name in classes:
    source_dir = source_root / class_name
    output_dir = output_root / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg"))

    print(f"{class_name}: {len(files)} originals")

    # Сохраняем оригиналы в augmented dataset
    counter = 1

    for file in files:
        img = Image.open(file).convert("RGB")
        img.save(output_dir / f"{class_name}_{counter:03d}.jpg", quality=95)
        counter += 1

    # Делаем augmented-картинки, пока не достигнем TARGET_COUNT
    while counter <= TARGET_COUNT:
        file = random.choice(files)
        img = Image.open(file).convert("RGB")

        # небольшое случайное вращение
        angle = random.uniform(-12, 12)
        img = img.rotate(angle, expand=False)

        # яркость
        brightness_factor = random.uniform(0.75, 1.25)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # контраст
        contrast_factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        # иногда зеркалим
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # иногда лёгкий blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

        # небольшой crop/zoom
        if random.random() < 0.5:
            width, height = img.size
            crop_percent = random.uniform(0.03, 0.10)

            left = int(width * crop_percent)
            top = int(height * crop_percent)
            right = int(width * (1 - crop_percent))
            bottom = int(height * (1 - crop_percent))

            img = img.crop((left, top, right, bottom))
            img = img.resize((width, height))

        img.save(
            output_dir / f"{class_name}_{counter:03d}.jpg",
            quality=95
        )

        counter += 1

    print(f"{class_name}: augmented to {TARGET_COUNT} images")

print("Done.")
