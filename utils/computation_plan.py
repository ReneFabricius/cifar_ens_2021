from abc import ABC, abstractmethod
import pandas as pd


class ComputationPlan(ABC):
    def __init__(self, plan: pd.DataFrame) -> None:
        self.plan_ = plan
    
    @abstractmethod
    def execute_evaluate_save(self) -> pd.DataFrame:
        pass
    
class ComputationPlanPWC(ComputationPlan):
    def execute_evaluate_save(self) -> pd.DataFrame:
        return pd.DataFrame()
    
class ComputationPlanCAL(ComputationPlan):
    def execute_evaluate_save(self) -> pd.DataFrame:
        return pd.DataFrame()
    