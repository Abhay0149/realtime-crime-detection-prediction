import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model


class ViolenceDetector:
    def __init__(self, model_path="violence.h5", frame_buffer_size=16):
        self.model = load_model(model_path)
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = deque(maxlen=frame_buffer_size)

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype("float32") / 255.0
        return frame

    def predict_violence(self, frames):
        if len(frames) != self.frame_buffer_size:
            return None, None

        frames_array = np.array(frames, dtype="float32")
        frames_array = np.expand_dims(frames_array, axis=0)

        prediction = self.model.predict(frames_array, verbose=0)
        print("Raw prediction:", prediction)

        # ✅ DEMO SAFE THRESHOLD (Very High)
        if prediction.shape[-1] == 1:
            prob = float(prediction[0][0])

            if prob >= 0.95:   # 🔥 High threshold
                return 1, prob
            else:
                return 0, prob

        else:
            probabilities = prediction[0]
            violence_prob = float(probabilities[1])

            if violence_prob >= 0.95:   # 🔥 High threshold
                return 1, probabilities
            else:
                return 0, probabilities


def process_frame(frame, detector):
    processed = detector.preprocess_frame(frame)
    detector.frame_buffer.append(processed)

    if len(detector.frame_buffer) == detector.frame_buffer_size:
        return detector.predict_violence(list(detector.frame_buffer))

    return None, None