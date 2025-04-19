from typing import Dict, Union
from pydantic import BaseModel

class ModelInput(BaseModel):
    features: Dict[str, Union[str, list, float, int]]