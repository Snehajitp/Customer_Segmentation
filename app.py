import warnings

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.config import (
    APP_HOST,
    APP_PORT,
    FEATURE_COLUMNS
)

warnings.filterwarnings("ignore")

app = FastAPI(title="Customer Categorizer")

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Load model only ONCE
# ---------------------------------------------------------

prediction_pipeline = PredictionPipeline()

# ---------------------------------------------------------
# Cluster Labels
# ---------------------------------------------------------

CLUSTER_NAMES = {
    0: "Budget Customer",
    1: "Regular Customer",
    2: "Premium Customer",
    3: "High Value Customer",
}

# ---------------------------------------------------------
# Form Helper
# ---------------------------------------------------------


class DataForm:

    def __init__(self, request: Request):
        self.request = request
        self.data = {}

    async def load(self):
        form = await self.request.form()

        self.data = {
            feature: form.get(feature)
            for feature in FEATURE_COLUMNS
        }

    def to_list(self):
        return [
            self.data[col]
            for col in FEATURE_COLUMNS
        ]


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "customer.html",
        {
            "request": request,
            "context": None,
        },
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):

    try:

        form = DataForm(request)
        await form.load()

        prediction = prediction_pipeline.predict(
            form.to_list()
        )

        cluster = int(prediction[0])

        return templates.TemplateResponse(
            "customer.html",
            {
                "request": request,
                "context": CLUSTER_NAMES.get(
                    cluster,
                    f"Cluster {cluster}",
                ),
            },
        )

    except Exception as e:

        return templates.TemplateResponse(
            "customer.html",
            {
                "request": request,
                "context": f"Prediction Failed: {str(e)}",
            },
        )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
    )