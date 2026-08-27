from fastapi import FastAPI
from pydantic import BaseModel, Field
from realtime_engine import RealtimePanoramaController  # Import từ file 2

app = FastAPI(title="PANORAMA Realtime HVAC Controller")
controller = RealtimePanoramaController()

class SensorInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, example=14)
    minute: int = Field(..., ge=0, le=59, example=15)
    temp_ext: float = Field(..., example=31.5)
    visitors_15m: int = Field(..., example=350)
    temp_int: float = Field(..., example=24.3)

@app.post("/api/v1/control-setpoint")
def get_control_setpoint(payload: SensorInput):
    return controller.process_15min_reading(
        current_hour=payload.hour,
        current_minute=payload.minute,
        temp_ext_15m=payload.temp_ext,
        visitors_15m=payload.visitors_15m,
        temp_int_15m=payload.temp_int
    )