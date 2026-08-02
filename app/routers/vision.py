from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import cv2
import base64
from app.schemas import FaceRecognitionResponse, ProductClassificationResponse
from app.services.cv_service import decode_image, grayscale, blur_image, detect_edges, detect_faces
from app.services.product_classifier import classify_product
from app.services.face_recognition_module import recognize_customer_face
from app.services.pipeline import pipeline
from app.routers.nlp import verify_api_key

router = APIRouter(
    tags=["Computer Vision"]
)

def image_to_base64(image):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode("utf-8")


@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face_endpoint(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    try:
        contents = await file.read()
        image = decode_image(contents)
        
        if not pipeline.face_extractor:
            raise HTTPException(status_code=500, detail="Face extractor model is not initialized.")
            
        result = recognize_customer_face(
            image=image,
            face_db=pipeline.face_db,
            extractor_model=pipeline.face_extractor
        )
        return {
            "filename": file.filename,
            **result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-product", response_model=ProductClassificationResponse)
async def classify_product_endpoint(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    if not pipeline.product_model:
        raise HTTPException(status_code=500, detail="Product classification model is not initialized.")
    try:
        contents = await file.read()
        image = decode_image(contents)
        
        # Call the existing service classifier
        result = classify_product(image)
        return {
            "filename": file.filename,
            **result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Backward Compatibility and CV Basics Utilities ---
@router.post("/vision/detect-face")
async def detect_face(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    try:
        contents = await file.read()
        image = decode_image(contents)
        faces = detect_faces(image)
        return {
            "filename": file.filename,
            "faces_detected": len(faces),
            "faces": faces
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/vision/preprocess")
async def preprocess_image(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    try:
        contents = await file.read()
        image = decode_image(contents)
        gray = grayscale(image)
        blurred = blur_image(image)
        edges = detect_edges(image)
        return {
            "filename": file.filename,
            "original_shape": list(image.shape),
            "grayscale_shape": list(gray.shape),
            "grayscale": image_to_base64(gray),
            "blurred": image_to_base64(blurred),
            "edges": image_to_base64(edges)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))