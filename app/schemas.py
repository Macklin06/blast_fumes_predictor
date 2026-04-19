from pydantic import BaseModel, Field

class BlastInput(BaseModel):
    heading_area_m2: float = Field(..., ge=9.0, le=25.0,
        description="Cross-sectional area of heading in m²")
    heading_length_m: float = Field(..., ge=30.0, le=150.0,
        description="Length of heading in metres")
    explosive_kg: float = Field(..., ge=30.0, le=150.0,
        description="Weight of explosive used in kg")
    fan_capacity_m3s: float = Field(..., ge=5.0, le=20.0,
        description="Auxiliary fan capacity in m³/s")
    duct_distance_from_face_m: float = Field(..., ge=5.0, le=25.0,
        description="Distance of duct from blast face in metres")
    temperature_c: float = Field(..., ge=25.0, le=38.0,
        description="Ambient temperature in °C")
    humidity_pct: float = Field(..., ge=60.0, le=95.0,
        description="Relative humidity percentage")

class PredictionOutput(BaseModel):
    predicted_clearance_time_mins: float
    model_predicted_time_mins: float
    regulatory_min_wait_mins: float
    dynamic_mandatory_wait_mins: float
    safety_recommendation: str

    model_config = {"json_schema_extra": {"example": {
        "predicted_clearance_time_mins": 45,
        "model_predicted_time_mins": 39,
        "regulatory_min_wait_mins": 30,
        "dynamic_mandatory_wait_mins": 45,
        "safety_recommendation": "WAIT — model suggests 39 minutes; enforce 45 minutes (mandatory dynamic minimum: 45, base floor: 30)."
    }}}