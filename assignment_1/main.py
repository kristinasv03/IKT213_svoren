import cv2
import numpy as np
from pathlib import Path

def print_image_information(image: np.ndarray) -> None:
    if image is None:
        print("Error: image is None.")
        return
    if image.ndim == 2:
        h, w = image.shape; c = 1
    elif image.ndim == 3:
        h, w, c = image.shape
    else:
        return
    print("Image Information")
    print(f"height: {h}")
    print(f"width: {w}")
    print(f"channels: {c}")
    print(f"size: {image.size}")
    print(f"dtype: {image.dtype}")

def save_camera_info(out_txt: Path) -> None:
    cam = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    _ = cam.read()  # warm-up
    fps = cam.get(cv2.CAP_PROP_FPS) or 0
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cam.release()
    with open(out_txt, "w") as f:
        f.write(f"fps: {int(fps) if fps == int(fps) else fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

def main():
    base = Path(__file__).parent
    # IV
    img = cv2.imread(str(base / "lena-1.png"), cv2.IMREAD_UNCHANGED)
    print_image_information(img)
    # V
    save_camera_info(base / "solutions" / "camera_outputs.txt")

if __name__ == "__main__":
    main()
