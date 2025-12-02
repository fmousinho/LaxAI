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
    learning_rate: Optional[float] = Field(default=0.0, description="Current learning rate")
    weight_decay: Optional[float] = Field(default=0.0, description="Weight decay value")

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
    


class EvalData(BaseModel):
    """Evaluation metrics data model."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    precision: Optional[float] = Field(default=0.0, description="Precision")
    recall: Optional[float] = Field(default=0.0, description="Recall")
    f1_score: Optional[float] = Field(default=0.0, description="F1 score")
    k1: Optional[float] = Field(default=0.0, description="Top-1 accuracy")
    k5: Optional[float] = Field(default=0.0, description="Top-5 accuracy")
    mean_avg_precision: Optional[float] = Field(default=0.0, description="Mean Average Precision")

    def __add__(self, other: 'EvalData') -> 'EvalData':
        """Add two EvalData instances for accumulation."""
        kwargs = {}
        for field in self.__fields__:
            kwargs[field] = self.__getattribute__(field) + other.__getattribute__(field)
        return EvalData(**kwargs)
        

    def __truediv__(self, scalar: float) -> 'EvalData':
        """Divide all evaluation metrics by a scalar for averaging."""
        if scalar == 0:
            raise ValueError("Cannot divide evaluation metrics by zero")
        kwargs = {}
        for field in self.__fields__:
            kwargs[field] = (self.__getattribute__(field)) / scalar
        return EvalData(**kwargs)

   
   