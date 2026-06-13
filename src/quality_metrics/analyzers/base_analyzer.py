from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, target_dir: str, files: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Analyzes code quality for a given list of files in the target directory.
        
        Args:
            target_dir: Absolute path of target codebase directory.
            files: List of absolute file paths to analyze.
            output_file: Optional path to save the output CSV.
            
        Returns:
            A pandas DataFrame containing quality metrics matching the unified schema.
        """
        pass
