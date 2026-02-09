from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_triage

app = FastAPI(title="Triage AI (Educational)", version="1.0")

class TriageInput(BaseModel):
    age: int
    hr: int
    sbp: int
    dbp: int
    resp_rate: int
    temp: float
    spo2: int
    pain: int
    chest_pain: int = 0
    sob: int = 0
    confusion: int = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: TriageInput):
    return predict_triage(inp.model_dump())