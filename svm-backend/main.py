# Model
from model import *
import pandas as pd
from collections import Counter
# API
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# OS File
import glob
import os

# Cấu hình
folder_path = "."
extension = "*.sav"
non_label_csv = "../phone_price_dataset/non-label.csv"
labeled_csv = "../phone_price_dataset/train.csv"

os.system("cls")

# Tải model
file_list = glob.glob(f"{folder_path}/{extension}")
if len(file_list) < 8:
    for file_path in file_list:
        os.remove(file_path)
    start_training_model(labeled_csv)
model_arr = load_model_from_file()

print("=====> API đang chạy ở http://localhost:8000")

# Khởi tạo web server
app = FastAPI()

# Mount thư mục tĩnh để phục vụ các tệp từ đường dẫn gốc "/"
# app.mount('/', StaticFiles(directory='dist', html=True))
# templates = Jinja2Templates(directory="dist")


# Cấu hình CORS
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tổng quan về dataset
@app.get("/api/info")
def models_list():
    df = pd.read_csv(labeled_csv)
    return JSONResponse(content=jsonable_encoder({
        "number_of_feature": df.shape[1] - 1,
        "number_of_sample": df.shape[0],
        "training_sample": int(df.shape[0]*0.75),
        "testing_sample": df.shape[0] - int(df.shape[0]*0.75),
    }))

# Danh sách các model
@app.get("/api/model")
def models_list():
    entries = []
    for item in model_arr:
        if "Sklearn" not in item["name"]:
            entry = item.copy()
            del entry["model"]
            entries.append(entry)
    return JSONResponse(content=jsonable_encoder({
        "entries": entries
    }))

# Lấy ngẫu nhiên danh sách từ tập dữ liệu chưa label
@app.get("/api/raw-data")
def models_list():
    raw_dataset=pd.read_csv(non_label_csv)
    raw_dataset = raw_dataset.drop("id", axis=1)
    sample = raw_dataset.sample(n=5)
    return JSONResponse(content=jsonable_encoder({
        "entries": sample.to_dict(orient='records')
    }))

# Dự đoán
@app.post("/api/predict")
async def process_data(data: dict):
    X_predict = pd.DataFrame.from_dict([data])
    entries = []
    predictions = []
    for item in model_arr:
        sample_prediction = item["model"].predict(X_predict)
        prediction = parse_prediction_to_text(sample_prediction)
        entry = {
            "name": item["name"],
            "accuracy": item["accuracy"],
            "predict_class": prediction["predict_class"],
            "score_prediction": prediction["score_prediction"]
        }
        entries.append(entry)
        predictions.append(prediction["predict_class"])
    counts = Counter(predictions)
    max_count = max(counts.values())
    most_common = [num for num, count in counts.items() if count == max_count]
    return JSONResponse(content=jsonable_encoder({
        "entries": entries,
        "best_choice": most_common[0]
    }))

# @app.get("/home")
# async def serve_home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})