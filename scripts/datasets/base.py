from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def load(self):
        """Load and return the dataset."""
        pass
