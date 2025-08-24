# iv_v_windows.py
import cv2
import numpy as np
from pathlib import Path

def print_image_information(image: np.ndarray) -> None:
    """
    Prints height, width, channels, size (# of values), and dtype of an image.
    """
    if image is None:
        print("Error: image is None. Check the path/filename.")
        return

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    elif image.ndim == 3:
        height, width, channels = image.shape
    else:
        print(f"Unexpected image ndim: {image.ndim}")
        return

    size = image.size
    dtype = image.dtype

    print("Image Information")
    print(f"height:   {height}")
    print(f"width:    {width}")
    print(f"channels: {channels}")
    print(f"size:     {size}")
    print(f"dtype:    {dtype}")

def main():
    base = Path(__file__).parent

    # --- IV: check lena.png (make sure filename matches yours!) ---
    img_path = base / "lena-1.png"   # change to "lena-1.png" if that's your actual file
    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    print_image_information(image)

    # --- V: camera info saved to solutions/camera_outputs.txt ---
    out_txt = base / "solutions" / "camera_outputs.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # Open camera with Windows backend (try MSMF first, fallback to DSHOW)
    cam = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cam.isOpened():
        # Could not open camera → write placeholders
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("fps: 0\nheight: 0\nwidth: 0\n")
        return

    # Warm-up read (helps some drivers report FPS/size correctly)
    _ = cam.read()

    fps = cam.get(cv2.CAP_PROP_FPS) or 0
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    cam.release()

    # Save results only to file
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"fps: {int(fps) if fps == int(fps) else fps}\n")
        f.write(f"height: {frame_height}\n")
        f.write(f"width: {frame_width}\n")

if __name__ == "__main__":
    main()
