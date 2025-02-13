import cv2
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class EnginePartsRecognition:
    def __init__(self, video_file, api_key):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.cap = cv2.VideoCapture(video_file if video_file else 0)  # 0 for webcam
        self.last_sent_time = 0  # Track last request time

        if not self.cap.isOpened():
            raise Exception("Error: Could not open video source.")

    def encode_image_to_base64(self, image):
        _, img_buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(img_buffer).decode('utf-8')

    def analyze_image_with_gemini(self, frame):
        if frame is None:
            return "No image available for analysis."

        image_data = self.encode_image_to_base64(frame)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """
                            Identify the engine part in this image and provide its names and what it made from only no details.
                            Engine_part_name|what it made from|
                            |----------------- |---------------- |
                            """
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    "description": "Engine part"
                }
            ])

        try:
            response = self.gemini_model.invoke([message])
            print("🔹 Gemini Response:", response.content)
        except Exception as e:
            print(f"Error invoking Gemini model: {e}")

    def process_video_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_sent_time >= 5:  # Send every 5 seconds
            self.last_sent_time = current_time
            threading.Thread(target=self.analyze_image_with_gemini, args=(frame,), daemon=True).start()

    def start_processing(self):
        window_name = "Engine Parts Recognition"
        cv2.namedWindow(window_name)

        while True:
            ret, frame = self.cap.read()
            frame=cv2.resize(frame,(800,500))
            if not ret:
                break

            cv2.imshow(window_name, frame)
            self.process_video_frame(frame)  # Only send every 5 seconds

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("🔹 Process completed.")


# Example usage:
if __name__ == "__main__":
    video_file = "parts.mp4"  # Use webcam if empty
    api_key = ""  # Replace with your actual API key
    processor = EnginePartsRecognition(video_file, api_key)
    processor.start_processing()
