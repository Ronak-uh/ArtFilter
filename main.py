import cv2
import mediapipe as mp
import numpy as np
import time  # NEW: Added for the 1-second stopwatch

# Initialize MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4
)

def apply_pop_art_effect(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pop_art = roi.copy()
    
    dark_mask = gray < 85
    mid_mask = (gray >= 85) & (gray < 170)
    light_mask = gray >= 170
    
    pop_art[dark_mask] = [200, 0, 255]
    pop_art[mid_mask] = [255, 255, 0]
    pop_art[light_mask] = [0, 255, 255]
    pop_art[::6, ::6] = [0, 0, 0]
    
    return pop_art

# NEW: The high-contrast comic book Black and White filter
def apply_bw_effect(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Sharp threshold for harsh comic-book shadows
    _, bw = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    
    # Convert back to 3 channels so OpenCV can composite it
    bw_art = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    
    # Halftone dots
    bw_art[::6, ::6] = [0, 0, 0]
    
    return bw_art

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    smooth_points = np.zeros((4, 2), dtype=np.float32)
    smoothing_factor = 0.6
    is_first_frame = True
    frames_lost = 0
    max_grace_frames = 3 

    # --- NEW: Gesture UI State Variables ---
    is_bw_mode = False          # Tracks which filter is active
    pinch_start_time = None     # Tracks when you started pinching
    pinch_triggered = False     # Prevents the filter from rapidly flashing on/off
    PINCH_THRESHOLD = 0.05      # How close the fingers need to be to register a pinch

    print("Press 'q' to quit.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            raw_points = []

            if detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
                hand_dict = {}
                
                for idx, handedness in enumerate(detection_result.handedness):
                    label = handedness[0].category_name
                    hand_dict[label] = detection_result.hand_landmarks[idx]

                if "Left" in hand_dict and "Right" in hand_dict:
                    left_hand = hand_dict["Left"]
                    right_hand = hand_dict["Right"]

                    # --- NEW: Pinch Detection Logic ---
                    # Calculate distance between Right Thumb (4) and Right Index (8)
                    dx = right_hand[8].x - right_hand[4].x
                    dy = right_hand[8].y - right_hand[4].y
                    pinch_distance = np.hypot(dx, dy)

                    # If fingers are touching (pinched)
                    if pinch_distance < PINCH_THRESHOLD:
                        if pinch_start_time is None:
                            pinch_start_time = time.time() # Start the stopwatch
                        elif not pinch_triggered and (time.time() - pinch_start_time >= 1.0):
                            # Stopwatch hit 1.0 seconds! Toggle the filter.
                            is_bw_mode = not is_bw_mode
                            pinch_triggered = True # Lock it so it only toggles once per pinch
                    else:
                        # Fingers separated. Reset the stopwatch and locks.
                        pinch_start_time = None
                        pinch_triggered = False
                    # ----------------------------------

                    raw_points = [
                        [int(left_hand[8].x * w), int(left_hand[8].y * h)],
                        [int(right_hand[8].x * w), int(right_hand[8].y * h)],
                        [int(right_hand[4].x * w), int(right_hand[4].y * h)],
                        [int(left_hand[4].x * w), int(left_hand[4].y * h)]
                    ]

            if len(raw_points) == 4:
                frames_lost = 0
                current_pts = np.array(raw_points)

                if is_first_frame:
                    smooth_points = current_pts.astype(np.float32)
                    is_first_frame = False
                else:
                    smooth_points += (current_pts - smooth_points) * smoothing_factor
            else:
                frames_lost += 1

            frame = cv2.convertScaleAbs(frame, alpha=0.4, beta=0)

            if frames_lost < max_grace_frames and not is_first_frame:
                poly_pts = np.int32(smooth_points).reshape((-1, 1, 2))
                x, y, bw_box, bh_box = cv2.boundingRect(poly_pts)

                pad = 50
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + bw_box + pad)
                y2 = min(h, y + bh_box + pad)

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2].copy()
                    local_poly_pts = poly_pts - np.array([[[x1, y1]]])

                    roi_h, roi_w = roi.shape[:2]
                    roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                    cv2.fillPoly(roi_mask, [local_poly_pts], 255)

                    bright_roi = cv2.convertScaleAbs(roi, alpha=2.5, beta=0) 
                    
                    # --- NEW: Apply the selected filter based on our gesture toggle ---
                    if is_bw_mode:
                        filtered_roi = apply_bw_effect(bright_roi)
                        glow_color = (255, 255, 255)       # Pure White Neon for B&W
                        inner_glow_color = (255, 255, 255)
                    else:
                        filtered_roi = apply_pop_art_effect(bright_roi)
                        glow_color = (255, 0, 255)         # Magenta Neon for CMYK
                        inner_glow_color = (255, 150, 255)
                    # ------------------------------------------------------------------

                    mask_3d = roi_mask[:, :, np.newaxis] == 255
                    roi = np.where(mask_3d, filtered_roi, roi)

                    glow_mask = np.zeros_like(roi)
                    cv2.polylines(glow_mask, [local_poly_pts], isClosed=True, color=glow_color, thickness=20)
                    glow_mask = cv2.GaussianBlur(glow_mask, (41, 41), 0)
                    
                    roi = cv2.add(roi, glow_mask)
                    cv2.polylines(roi, [local_poly_pts], isClosed=True, color=inner_glow_color, thickness=3)

                    frame[y1:y2, x1:x2] = roi

            elif frames_lost >= max_grace_frames:
                is_first_frame = True

            cv2.imshow("Hand-Framed Dynamic Pop Art", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()