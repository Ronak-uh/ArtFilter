#  ArtFilter

A real-time augmented reality computer vision application that tracks your hand gestures to create a dynamic, glowing neon frame containing comic-book-style Pop-Art filters. Built with Python, OpenCV, and Google's MediaPipe.

![Demo Placeholder](https://via.placeholder.com/800x450.png?text=Replace+with+a+GIF+or+Screenshot+of+your+filter+in+action!)

##  Features

*   **Dynamic Hand Tracking:** Uses MediaPipe's Tasks API to track your index fingers and thumbs in real-time, mapping a custom, free-form polygon that twists and morphs with your physical hand movements.
*   **Vectorized Image Processing:** Implements heavy image filters (posterization, halftone dot-grids) using highly optimized NumPy array slicing rather than slow `for` loops.
*   **Gesture-Based UI Toggle:** Features a built-in gesture control engine. Pinch and hold your right thumb and index finger for 1.0 second to seamlessly toggle between a vibrant CMYK Pop-Art filter and a high-contrast Black & White comic style.
*   **Neon Bloom Compositing:** Utilizes OpenCV's Gaussian blur and image addition to render a realistic, glowing neon border around the dynamic frame.
*   **ROI Optimization:** Calculates a strict Region of Interest (ROI) bounding box on every frame to apply heavy effects *only* where needed, dramatically increasing rendering FPS.

##  Tech Stack

*   **Python 3.x**
*   **[OpenCV](https://opencv.org/):** Real-time video capture, image manipulation, and visual compositing.
*   **[MediaPipe](https://developers.google.com/mediapipe):** State-of-the-art AI skeletal hand tracking.
*   **[NumPy](https://numpy.org/):** Complex matrix math and localized array transformations.

##  Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/Ronak-uh/ArtFilter.git
cd ArtFilter
```

**2. Create and activate a virtual environment (Optional but recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

**3. Install dependencies**

```bash
pip install opencv-python mediapipe numpy
```

**4. Download the MediaPipe Hand Landmarker Model**
The modern MediaPipe Tasks API requires the model weight file to be downloaded locally. Run this command to download it directly into the project directory:

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**5. Run the application**

```bash
python main.py
```

##  How to Use

1. Stand in front of your webcam in a reasonably well-lit room.
2. Hold up both hands and point your **Index Fingers** up and your **Thumbs** inward (like a movie director framing a shot).
3. Move, rotate, or cross your hands to watch the AR frame dynamically warp to connect your fingertips.
4. **The Secret Toggle:** Pinch your **Right Index Finger** and **Right Thumb** together and hold them for exactly 1 second to switch the art style to Black & White! Un-pinch and pinch again to switch back.
5. Press **`q`** on your keyboard to exit the application.

---
