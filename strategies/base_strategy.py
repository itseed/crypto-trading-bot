from abc import ABC, abstractmethod
import pandas as pd
from typing import Union

class BaseStrategy(ABC):
    @abstractmethod
    def signal(self, df: pd.DataFrame) -> Union[str, None]:
        pass