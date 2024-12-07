from pydantic import BaseModel

class bankdata(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float