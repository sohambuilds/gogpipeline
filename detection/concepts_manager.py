import json
from pathlib import Path
import importlib.util
def load_protected_concepts(json_path: str):
    """
    Loads the protected concepts JSON from disk and returns a list of dictionaries:
    [
      {
        "concept": <str>,
        "synonyms": <list of str>,
        "description": <str>
      },
      ...
    ]
    """
    path_obj = Path(json_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Protected concepts JSON not found at: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_test_prompts(py_path: str):
    spec = importlib.util.spec_from_file_location("test_prompts", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.test_prompts
