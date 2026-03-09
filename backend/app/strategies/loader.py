"""Load strategy classes from Python files."""
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Type
from .base import BaseStrategy


def load_strategy_from_file(file_path: str) -> Type[BaseStrategy]:
    """
    Load a strategy class from a Python file.
    
    The file should define a class that inherits from BaseStrategy.
    The class name should end with 'Strategy' (e.g., MyStrategy, TrendStrategy).
    
    Args:
        file_path: Path to the Python file containing the strategy
        
    Returns:
        Strategy class (not instance)
        
    Raises:
        ValueError: If no strategy class is found or multiple classes found
        ImportError: If the file cannot be imported
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")
    
    if not file_path.suffix == ".py":
        raise ValueError(f"Strategy file must be a Python file (.py): {file_path}")
    
    # Load the module
    module_name = f"strategy_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Find strategy classes (subclasses of BaseStrategy, excluding BaseStrategy itself)
    strategy_classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseStrategy)
            and obj is not BaseStrategy
        ):
            strategy_classes.append(obj)
    
    if len(strategy_classes) == 0:
        raise ValueError(
            f"No strategy class found in {file_path}. "
            "Define a class that inherits from BaseStrategy."
        )
    
    if len(strategy_classes) > 1:
        raise ValueError(
            f"Multiple strategy classes found in {file_path}: {[c.__name__ for c in strategy_classes]}. "
            "Only one strategy class per file is allowed."
        )
    
    return strategy_classes[0]


def load_strategy_from_module(module_path: str, class_name: Optional[str] = None) -> Type[BaseStrategy]:
    """
    Load a strategy class from a module path.
    
    Args:
        module_path: Python module path (e.g., 'app.strategies.my_strategy')
        class_name: Name of the strategy class (if None, auto-detect)
        
    Returns:
        Strategy class (not instance)
    """
    module = importlib.import_module(module_path)
    
    if class_name:
        if not hasattr(module, class_name):
            raise ValueError(f"Class {class_name} not found in module {module_path}")
        strategy_class = getattr(module, class_name)
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{class_name} is not a subclass of BaseStrategy")
        return strategy_class
    
    # Auto-detect strategy class
    strategy_classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseStrategy)
            and obj is not BaseStrategy
        ):
            strategy_classes.append(obj)
    
    if len(strategy_classes) == 0:
        raise ValueError(f"No strategy class found in module {module_path}")
    
    if len(strategy_classes) > 1:
        raise ValueError(
            f"Multiple strategy classes found in {module_path}: {[c.__name__ for c in strategy_classes]}. "
            "Specify class_name parameter."
        )
    
    return strategy_classes[0]
