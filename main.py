from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import torch
from torchvision import transforms, models
from PIL import Image
from typing import List
from io import BytesIO

app = FastAPI()

# ตั้งค่า static files และ templates folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join('Model/model.pth')

def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

if os.path.exists(MODEL_PATH):
    model, device = load_trained_model()
else:
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index.html", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/contact.html", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, images: List[UploadFile] = File(...)):
    results = []
    for image in images:
        if image.filename == '':
            continue
        try:
            contents = await image.read()
            img = Image.open(BytesIO(contents)).convert("RGB")

            # บันทึกไฟล์ภาพ (optional)
            save_path = os.path.join(UPLOAD_FOLDER, image.filename)
            with open(save_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            results.append((image.filename, f"Invalid image file: {e}"))
            continue

        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        label = 'MIBC' if predicted.item() == 0 else 'NMIBC'
        results.append((image.filename, label))

    return templates.TemplateResponse("contact.html", {"request": request, "results": results})
