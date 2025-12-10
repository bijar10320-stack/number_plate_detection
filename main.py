import cv2
import torch
import easyocr
from ultralytics import YOLO


class NumberPlateDetector:


    def __init__(self, model_path: str, ocr_langs=None, use_gpu=True):
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        print(f"Using device: {self.device}")


        self.model = YOLO("number_plate_detection/best.pt").to(self.device)


        ocr_langs = ocr_langs or ['en']
        self.reader = easyocr.Reader(ocr_langs, gpu=(self.device == "cuda"))

        self.seen_plates = set()

    def detect_and_ocr(self, frame):

        results = self.model(frame, device=self.device)

        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)


                crop = frame[y1:y2, x1:x2]


                ocr_out = self.reader.readtext(crop)

                plate_text = None
                if len(ocr_out) > 0:
                    plate_text = ocr_out[0][1]
                    if plate_text not in self.seen_plates:
                        print(f"[DETECTED] {plate_text}")
                        self.seen_plates.add(plate_text)

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "text": plate_text
                })

        return detections

    def annotate_frame(self, frame, detections):

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            text = det["text"]


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


            if text:
                cv2.putText(
                    frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

        return frame

    def process_video(self, input_video: str, output_video="output_anpr.mp4", skip_frames=5):


        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(" Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video loaded: {width}x{height} @ {fps} FPS")


        out_width, out_height = 1920, 1080
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (out_width, out_height))

        frame_idx = 0

        print("Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip_frames != 0:
                continue


            small = cv2.resize(frame, (1280, 720))
            results = self.model(small, device=self.device)

            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:

                    scale_x = frame.shape[1] / 1280
                    scale_y = frame.shape[0] / 720
                    x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)

                    crop = frame[y1:y2, x1:x2]
                    ocr_output = self.reader.readtext(crop)

                    plate_text = None
                    if len(ocr_output) > 0:
                        plate_text = ocr_output[0][1]

                        if plate_text not in self.seen_plates:
                            print(f"[DETECTED] {plate_text}")
                            self.seen_plates.add(plate_text)

                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "text": plate_text
                    })

            annotated = self.annotate_frame(frame, detections)


            output_resized = cv2.resize(annotated, (out_width, out_height))
            out.write(output_resized)

        cap.release()
        out.release()
        print(f" Processing complete Saved as {output_video}")


if __name__ == "__main__":
    detector = NumberPlateDetector(
        model_path="best.pt",
        ocr_langs=['en'],
        use_gpu=True
    )

    detector.process_video(
        input_video="input.mp4",
        output_video="output_anpr.mp4",
        skip_frames=5
    )

