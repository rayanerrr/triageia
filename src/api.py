import json
import logging
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from predict import predict_triage, META_PATH, LOG_PATH

logger = logging.getLogger(__name__)

app = FastAPI(title="Triage AI", version="2.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur."}
    )


# --- Input validation with Pydantic ---
class TriageInput(BaseModel):
    age: int = Field(ge=0, le=120)
    hr: int = Field(ge=20, le=250)
    sbp: int = Field(ge=40, le=300)
    dbp: int = Field(ge=20, le=200)
    resp_rate: int = Field(ge=4, le=70)
    temp: float = Field(ge=30.0, le=45.0)
    spo2: int = Field(ge=50, le=100)
    pain: int = Field(ge=0, le=10)
    chest_pain_severity: int = Field(default=0, ge=0, le=3)
    sob_severity: int = Field(default=0, ge=0, le=3)
    confusion: int = Field(default=0, ge=0, le=1)
    comorbidity_cardiac: int = Field(default=0, ge=0, le=1)
    comorbidity_respiratory: int = Field(default=0, ge=0, le=1)
    comorbidity_diabetes: int = Field(default=0, ge=0, le=1)

    @model_validator(mode='after')
    def check_bp_consistency(self):
        if self.dbp >= self.sbp:
            raise ValueError(
                f"PAD ({self.dbp}) doit etre inferieure a PAS ({self.sbp})"
            )
        return self


# --- Routes ---
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(inp: TriageInput):
    data = inp.model_dump()
    data["num_comorbidities"] = (
        data["comorbidity_cardiac"]
        + data["comorbidity_respiratory"]
        + data["comorbidity_diabetes"]
    )
    return predict_triage(data)


@app.get("/metrics")
def metrics():
    """Expose model metadata and performance metrics."""
    if not META_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "meta.json introuvable. Entrainez le modele d'abord."}
        )
    with open(META_PATH, "r") as f:
        return json.load(f)


@app.get("/history")
def history(limit: int = Query(default=20, ge=1, le=200)):
    """Return the last N predictions from the log."""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().split("\n")
    lines = [l for l in lines if l.strip()]
    recent = lines[-limit:]
    recent.reverse()
    results = []
    for line in recent:
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results

