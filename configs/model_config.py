from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "microsoft/phi-2"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    device_map: dict = None
    
    def __post_init__(self):
        if self.device_map is None:
            self.device_map = {"": 0}
