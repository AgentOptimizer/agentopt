"""
Core model selection functionality.
"""
from typing import Optional, Callable, Dict, List, Any, Tuple, Union
import json
import time
from pathlib import Path
import pandas as pd


class ModelProxy:
    """
    Proxy that forwards attribute access to the current model.
    Stores the model directly in _model attribute.
    """

    def __init__(self, attr_name: str, initial_model: Any = None) -> None:
        self.attr_name = attr_name
        self._model = initial_model  # Store model directly

    def __getattr__(self, name):
        if self._model is None:
            raise RuntimeError(
                f"No model set for attribute '{self.attr_name}'. "
                f"Call bind_model(obj, '{self.attr_name}', model) first."
            )
        return getattr(self._model, name)


def bind_model(obj: Any, attr_path: str, model: Any) -> Any:
    """
    Bind a model to an object's attribute, supporting nested paths.
    
    The attr_path can be:
    - A simple attribute: "model" → obj.model
    - A nested path: "B.C" → obj.B.C
    - A nested path with leading dot: ".B.C" → obj.B.C (dot is stripped)
    
    For LangChain agents created with ModelProxy, updates the existing proxy.
    For other objects, creates/updates a ModelProxy at the specified path.
    
    Usage:
        # Simple attribute
        bind_model(agent, "model", candidate_model)
        
        # Nested attribute
        bind_model(agent, "B.C", candidate_model)  # Sets agent.B.C
        
        # For model selection, rebind with different models:
        for model in candidate_models:
            bind_model(agent, "B.C", model)
            result = agent.invoke(...)
    
    Args:
        obj: Object to bind model to
        attr_path: Path to the attribute (e.g., "model", "B.C", ".B.C")
        model: The model object to bind
    """
    # Strip leading dot if present
    attr_path = attr_path.lstrip('.')
    
    # Split path into parts
    parts = attr_path.split('.')
    
    # Navigate to the target object (all parts except the last)
    target_obj = obj
    path_so_far = []
    for part in parts[:-1]:
        path_so_far.append(part)
        if not hasattr(target_obj, part):
            current_path = '.'.join(path_so_far)
            raise AttributeError(
                f"Cannot bind model: path '{attr_path}' is invalid. "
                f"Object has no attribute '{part}' at '{current_path}'"
            )
        target_obj = getattr(target_obj, part)
    
    # Get the final attribute name
    final_attr = parts[-1]
    
    # Check if attribute already exists and is a ModelProxy
    if hasattr(target_obj, final_attr):
        existing = getattr(target_obj, final_attr)
        if isinstance(existing, ModelProxy):
            # Update existing proxy directly
            existing._model = model
            return obj
    
    # Create new proxy and set it as attribute
    proxy = ModelProxy(attr_path, model)
    setattr(target_obj, final_attr, proxy)
    return obj


# [TODO: Temporary] Extract dataset loading function outside of class
def load_dataset(dataset_dir: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load evaluation tasks from dataset directory.
    
    Looks for JSONL files in the dataset_dir and loads tasks.
    Expected format: JSONL with "question" and "output" fields.
    Each line is a JSON object: {"question": "...", "output": "..."}
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        List of evaluation tasks (only first one for quick testing)
    """
    if not dataset_dir:
        return []
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Look for JSONL files
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in dataset directory: {dataset_dir}")
    
    # Load first JSONL file (can be extended to load multiple)
    tasks = []
    with open(jsonl_files[0], 'r', encoding='utf-8') as f:
        # Only load first line for quick testing
        first_line = f.readline().strip()
        if first_line:
            item = json.loads(first_line)
            
            task = {}
            # Convert question to LangChain messages format
            if "question" in item:
                task["messages"] = [{"role": "user", "content": item["question"]}]
            else:
                raise ValueError(f"Missing 'question' field in dataset entry: {item}")
            
            # Store expected answer from "output" field
            if "output" in item:
                task["expected_answer"] = item["output"]
            else:
                raise ValueError(f"Missing 'output' field in dataset entry: {item}")
            
            tasks.append(task)
    
    return tasks


class ModelSelector:
    """
    Selects the best model for an agent by evaluating on a dataset.
    """
    
    def __init__(
        self,
        agent: Any,
        models: Dict[str, List[Union[str, Any]]],
        accuracy_fn: Callable[[str, str], bool],
        dataset_dir: Optional[str] = None,
    ):
        """
        Initialize the model selector.
        
        Args:
            agent: The agent instance to optimize
            models: Dictionary mapping attribute names to list of model objects or model name strings
                    e.g., {".model": [ChatOpenAI(...), "openai/gpt-4o"]}
                    String names will be automatically converted to model objects.
            accuracy_fn: Function that takes (expected_answer, actual_output) and returns True if correct
            dataset_dir: Optional path to evaluation dataset directory
        """
        from .model_factory import normalize_models
        
        self.agent = agent
        self.accuracy_fn = accuracy_fn
        self.dataset_dir = dataset_dir
        
        # Normalize models: convert strings to objects if needed
        self.models = normalize_models(models)
        
        # Store model objects
        self._model_objects = {}
        for attr_name, model_objects in self.models.items():
            self._model_objects[attr_name] = model_objects
    
    def select_best(
        self,
        evaluation_tasks: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """
        Select the best model for each attribute.
        
        Args:
            evaluation_tasks: Optional list of tasks to evaluate on.
                             If None and dataset_dir is provided, loads from dataset_dir.
                             If both None, uses a default evaluation.
        
        Returns:
            DataFrame with columns: model_name, accuracy, latency_seconds, attribute, best
        """
        # Load tasks from dataset_dir if not provided
        if evaluation_tasks is None and self.dataset_dir:
            evaluation_tasks = load_dataset(self.dataset_dir)
        
        all_results = []
        
        for attr_name, model_objects in self._model_objects.items():
            # attr_name can be like ".model", "model", "B.C", ".B.C", etc.
            # bind_model handles stripping the leading dot and parsing nested paths
            
            best_model = None
            best_score = float('inf')
            best_model_name = None
            
            # Display name (strip leading dot for readability)
            display_name = attr_name.lstrip('.')
            
            print(f"\n{'='*60}")
            print(f"Selecting best model for attribute: {display_name}")
            print(f"{'='*60}\n")
            
            for model_obj in model_objects:
                # Get model name for display (try to extract from model object)
                model_name = self._get_model_name(model_obj)
                
                # Bind model to agent (attr_name can be nested like "B.C")
                bind_model(self.agent, attr_name, model_obj)
                
                try:
                    # Evaluate model (returns accuracy and latency)
                    accuracy, latency = self._evaluate_model(model_obj, evaluation_tasks)
                    
                    print(f"✓ Model: {model_name}, Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s")
                    
                    all_results.append({
                        "model_name": model_name,
                        "accuracy": accuracy,
                        "latency_seconds": latency,
                        "attribute": display_name,
                    })
                    
                    # For backward compatibility, still track best model
                    # Lower score = higher accuracy (1 - accuracy)
                    score = 1 - accuracy
                    if score < best_score:
                        best_score = score
                        best_model = model_obj
                        best_model_name = model_name
                        
                except Exception as e:
                    print(f"✗ Model {model_name} failed: {e}")
                    all_results.append({
                        "model_name": model_name,
                        "accuracy": 0.0,
                        "latency_seconds": 0.0,
                        "attribute": display_name,
                    })
                    continue
            
            if best_model_name:
                # Bind the best model
                bind_model(self.agent, attr_name, best_model)
                print(f"\n🏆 Best model for {display_name}: {best_model_name} (accuracy: {1-best_score:.2%})")
            else:
                print(f"\n✗ No models succeeded for {display_name}")
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Mark best model for each attribute
        if not df.empty:
            df['best'] = False
            for attr in df['attribute'].unique():
                attr_df = df[df['attribute'] == attr]
                best_idx = attr_df['accuracy'].idxmax()
                df.loc[best_idx, 'best'] = True
        
        return df
    
    def _get_model_name(self, model_obj: Any) -> str:
        """Extract model name from model object for display purposes."""
        # Try common attributes for model name
        if hasattr(model_obj, 'model_name'):
            return str(model_obj.model_name)
        elif hasattr(model_obj, 'model'):
            return str(model_obj.model)
        elif hasattr(model_obj, '__class__'):
            return model_obj.__class__.__name__
        else:
            return str(model_obj)
    
    def _evaluate_model(
        self,
        model_obj: Any,
        evaluation_tasks: Optional[List[Any]] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate a model using accuracy metric and measure latency.
        
        Args:
            model_obj: The model object being evaluated
            evaluation_tasks: List of tasks to evaluate on (with expected_answer if available)
        
        Returns:
            Tuple of (accuracy, latency_seconds)
        """
        if evaluation_tasks is None:
            # Load from dataset_dir if available
            if self.dataset_dir:
                evaluation_tasks = load_dataset(self.dataset_dir)
            else:
                # Default evaluation: single task
                evaluation_tasks = [
                    {"messages": [{"role": "user", "content": "what is 2 + 2?"}], "expected_answer": "4"}
                ]
        
        # Run agent on all tasks and collect results
        correct = 0
        total = len(evaluation_tasks)
        total_latency = 0.0
        
        for task in evaluation_tasks:
            try:
                # Measure latency
                start_time = time.time()
                # agent.invoke expects a dict with "messages" key
                result = self.agent.invoke({"messages": task["messages"]})
                latency = time.time() - start_time
                total_latency += latency
                
                # Extract expected answer
                expected_answer = task.get("expected_answer", "")
                
                # Extract actual answer from result
                # Result can be a dict with 'messages' or a string
                if isinstance(result, dict) and "messages" in result:
                    # Get last message content (LangChain message objects have .content attribute)
                    messages = result["messages"]
                    if messages:
                        last_message = messages[-1]
                        # Handle both dict-like and object-like message formats
                        if hasattr(last_message, 'content'):
                            actual_answer = str(last_message.content)
                        elif isinstance(last_message, dict):
                            actual_answer = str(last_message.get("content", ""))
                        else:
                            actual_answer = str(last_message)
                    else:
                        actual_answer = ""
                else:
                    actual_answer = str(result)
                
                # Check if answer is correct using the provided accuracy function
                if expected_answer:
                    if self.accuracy_fn(expected_answer, actual_answer):
                        correct += 1
                else:
                    # No expected answer, just check if we got a response
                    if actual_answer:
                        correct += 1
                        
            except Exception as e:
                # If task fails, count as incorrect
                pass
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0
        
        return accuracy, avg_latency
    
