from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class MetricsData(BaseModel):
    """Metrics data model for training statistics."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss: Optional[float] = Field(default=0.0, description="Triplet loss")
    gradient_norm: Optional[float] = Field(default=0.0, description="Gradient norm")
    positive_distance: Optional[float] = Field(default=0.0, description="Average positive pair distance")
    negative_distance: Optional[float] = Field(default=0.0, description="Average negative pair distance")
    mining_hard: Optional[float] = Field(default=0.0, description="Ratio of hard triplets")
    mining_semi_hard: Optional[float] = Field(default=0.0, description="Ratio of semi-hard triplets")
    mining_easy: Optional[float] = Field(default=0.0, description="Ratio of easy triplets")
    model_variance: Optional[float] = Field(default=0.0, description="Embedding variance")

    def __add__(self, other: 'MetricsData') -> 'MetricsData':
        """Add two MetricsData instances for accumulation."""
        kwargs = {}
        for field in self.__fields__:
            kwargs[field] = self.__getattribute__(field) + other.__getattribute__(field)
        return MetricsData(**kwargs)
        

    def __truediv__(self, scalar: float) -> 'MetricsData':
        """Divide all metrics by a scalar for averaging."""
        if scalar == 0:
            raise ValueError("Cannot divide metrics by zero")
        kwargs = {}
        for field in self.__fields__:
            kwargs[field] = (self.__getattribute__(field)) / scalar
        return MetricsData(**kwargs)
    

   
   