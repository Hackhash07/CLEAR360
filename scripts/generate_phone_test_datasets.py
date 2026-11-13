"""
scripts/generate_phone_test_datasets.py — Generate Smartphone Photo Datasets
============================================================================

Generates structured test datasets representing real smartphone photography:
  1. data/phone_datasets/orbital_4photos/: 4 sequential orbital photos with ~70% overlap
  2. data/phone_datasets/orbital_12photos/: 12 sequential orbital photos circling 360°
  3. data/phone_datasets/surround_4photos/: 4 outward-facing surround photos (0% mutual overlap)

Embeds authentic EXIF tags into all rendered JPEG files (Apple iPhone 14 Pro, 26mm lens).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ExifTags

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def embed_iphone_exif(img_path: Path, focal_35mm: int = 26) -> None:
    """Embed authentic smartphone EXIF tags into a JPEG image."""
    with Image.open(img_path) as img:
        exif = img.getexif()
        # Standard EXIF tags
        exif[0x010F] = "Apple"               # Make
        exif[0x0110] = "iPhone 14 Pro"       # Model
        exif[0x0131] = "iOS 17.5"            # Software
        exif[0x920A] = (686, 100)            # FocalLength = 6.86 mm
        exif[0xA405] = int(focal_35mm)       # FocalLengthIn35mmFilm = 26 mm
        exif[0x829D] = (178, 100)            # FNumber = 1.78
        exif[0x8827] = 64                    # ISO = 64
        exif[0x9003] = "2026:11:13 15:30:00" # DateTimeOriginal

        # Save back with EXIF
        img.save(img_path, "JPEG", quality=95, exif=exif)


def generate_datasets():
    import subprocess
    import shutil

    data_dir = PROJECT_ROOT / "data" / "phone_datasets"
    dir_4orb = data_dir / "orbital_4photos"
    dir_12orb = data_dir / "orbital_12photos"
    dir_4sur = data_dir / "surround_4photos"

    for d in [dir_4orb, dir_12orb, dir_4sur]:
        d.mkdir(parents=True, exist_ok=True)

    blend_file = PROJECT_ROOT / "sim" / "blender_scene.blend"
    if not blend_file.exists():
        print(f"[ERROR] Blend file not found: {blend_file}")
        sys.exit(1)

    print("[CLEAR360] Rendering orbital smartphone photos using Blender...")

    # Python script executed inside Blender to render camera viewpoints
    blender_script = f"""
import bpy, mathutils, math
from pathlib import Path

blend_path = "{blend_file}"
dir_4orb = Path("{dir_4orb}")
dir_12orb = Path("{dir_12orb}")

# Setup camera
cam_data = bpy.data.cameras.new("PhoneCamera")
cam_data.lens = 26
cam_data.sensor_width = 36
cam_obj = bpy.data.objects.new("PhoneCamera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1280
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "JPEG"

# 1. 4-Photo Orbital Sequence (Front-left arc, ~70% overlap between adjacent shots)
positions_4 = [
    (1.8, 3.8, 1.4),
    (0.9, 4.1, 1.4),
    (-0.2, 4.2, 1.4),
    (-1.4, 3.9, 1.4),
]
target = mathutils.Vector((0.0, 1.8, 0.6))

for idx, pos in enumerate(positions_4):
    cam_obj.location = mathutils.Vector(pos)
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    out_file = dir_4orb / f"IMG_{{idx+1:04d}}.jpg"
    scene.render.filepath = str(out_file)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered 4-photo {{idx+1}}/4: {{out_file.name}}")

# 2. 12-Photo 360° Orbital Sequence (~70% overlap around entire Creta SUV)
num_12 = 12
radius = 4.2
height = 1.45
center = mathutils.Vector((0.0, 0.0, 0.6))

for i in range(num_12):
    angle_rad = i * (2.0 * math.pi / num_12)
    px = radius * math.cos(angle_rad)
    py = radius * math.sin(angle_rad)
    cam_obj.location = mathutils.Vector((px, py, height))
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    out_file = dir_12orb / f"IMG_{{i+1:04d}}.jpg"
    scene.render.filepath = str(out_file)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered 12-photo {{i+1}}/12: {{out_file.name}}")
"""

    cmd = ["blender", "-b", str(blend_file), "--python-expr", blender_script]
    subprocess.run(cmd, check=True)

    # 3. Copy 4 Outward-Facing Surround Photos (Failure Test Case: 0% mutual overlap)
    surround_src = PROJECT_ROOT / "data" / "photoreal_frames"
    cams = ["front", "rear", "left", "right"]
    for idx, c in enumerate(cams):
        src_img = surround_src / c / "frame_0000.png"
        if src_img.exists():
            dest_jpg = dir_4sur / f"IMG_SURROUND_{idx+1:02d}_{c.upper()}.jpg"
            with Image.open(src_img) as im:
                im.convert("RGB").save(dest_jpg, "JPEG", quality=95)

    # Embed EXIF metadata in all rendered datasets
    print("[CLEAR360] Embedding iPhone 14 Pro EXIF metadata into photos...")
    for d in [dir_4orb, dir_12orb, dir_4sur]:
        for f in d.glob("*.jpg"):
            embed_iphone_exif(f, focal_35mm=26)

    print(f"[CLEAR360] Generated 4-photo orbital dataset: {len(list(dir_4orb.glob('*.jpg')))} photos")
    print(f"[CLEAR360] Generated 12-photo orbital dataset: {len(list(dir_12orb.glob('*.jpg')))} photos")
    print(f"[CLEAR360] Generated 4-photo surround dataset: {len(list(dir_4sur.glob('*.jpg')))} photos")


if __name__ == "__main__":
    generate_datasets()

