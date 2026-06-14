import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
from agent import get_recycling_classification

app = FastAPI(title="Yalla Khadra - Waste Classification API")

# YOLO Model 
MODEL_PATH = "Model/transfer_400_acc-91.15_model_epochs-100_acc-97.14.pt"
model = YOLO(str(MODEL_PATH))


@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 3. Run inference using YOLOv8
        results = model(image)

        # 4. Extract classification results
        result = results[0]

        top_class_idx = result.probs.top1
        top_class_name = result.names[top_class_idx]
        top_confidence = float(result.probs.top1conf)

        # 5. Get AI recycling classification from GPT-4.1
        ai_result = get_recycling_classification(top_class_name)

        return {
            "prediction": top_class_name,
            "confidence": round(top_confidence, 4),
            "AIIsRecyclable": ai_result.get("AIIsRecyclable", False),
            "AIExplanation": ai_result.get("AIExplanation", "N/A"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

