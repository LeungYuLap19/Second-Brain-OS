import yaml
import glob
import os
from .paths import LLM_MODEL_FILE, SYSTEM_PROMPT_FILE

class Settings:
  """
  Singleton Pattern for Settings

  This class ensures that configuration files (YAML and prompt text) are loaded
  only once throughout the application lifecycle, providing a single source of
  truth for all settings.
  """
  _instance = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._initialize()
    return cls._instance

  def _load_yaml(self, path: str):
    if not os.path.isfile(path):
      raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
      return yaml.safe_load(file)

  def _load_system_prompt(self, path: str):
    if not os.path.isfile(path):
      raise FileNotFoundError(f"Prompt file not found: {path}")

    try:
      with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()
    except Exception as e:
      raise IOError(f"Failed to read prompt file {path}: {e}")

  def _initialize(self):
    try:
      self.llm_model = self._load_yaml(LLM_MODEL_FILE)
      self.system_prompt = self._load_system_prompt(SYSTEM_PROMPT_FILE)
    except Exception as e:
      print(f"Error loading settings: {e}")
      raise

  def get_base_url(self):
    return self.llm_model.get("OLLAMA_BASE_URL")

  def get_embedding_model(self):
    return self.llm_model.get("EMBEDDING_MODEL")

  def get_llm_model_config(self):
    return self.llm_model.get("LLM_MODEL")

  def get_system_prompt(self):
    return self.system_prompt

settings = Settings()