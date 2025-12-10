from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid
from number_plate_detector import NumberPlateDetector

app = FastAPI()


OUTPUT_DIR = "processed_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


detector = NumberPlateDetector(
    model_path="number_plate_detection/best.pt",
    ocr_langs=['en'],
    use_gpu=True
)


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):

    file_ext = file.filename.split(".")[-1]
    temp_input_path = f"temp_{uuid.uuid4()}.{file_ext}"

    with open(temp_input_path, "wb") as f:
        f.write(await file.read())


    output_path = os.path.join(OUTPUT_DIR, f"processed_{uuid.uuid4()}.mp4")


    detector.process_video(temp_input_path, output_video=output_path, skip_frames=5)

    os.remove(temp_input_path)


    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="processed_video.mp4"
    )