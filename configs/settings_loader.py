import yaml
import glob
import os

BASE_CONFIG_DIR = "configs"

"""
  For loading:
  - ollama_models.yaml
  - agent_config.yaml
"""
def load_yaml(path: str):
  with open(path, "r", encoding="utf-8") as file:
    return yaml.safe_load(file)

"""Get all agent prompts"""
def load_system_prompts():
  prompts_dir = os.path.join(BASE_CONFIG_DIR, "system_prompts")
  prompt_files = glob.glob(os.path.join(prompts_dir, "*.txt"))

  prompts = {}
  for path in prompt_files:
    key = os.path.basename(path).replace("_prompt.txt", "")
    with open(path, "r", encoding="utf-8") as file:
      prompts[key] = file.read()

  return prompts

class Settings:
  """Singleton-like class to load all project configuration settings."""
  
  def __init__(self):
    self.ollama_models = load_yaml(os.path.join(BASE_CONFIG_DIR, "ollama_models.yaml"))
    self.system_prompts = load_system_prompts()

  def get_base_url(self):
    return self.ollama_models["OLLAMA_BASE_URL"]

  def get_embedding_model(self):
    return self.ollama_models["EMBEDDING_MODEL"]

  def get_agent_model_config(self, agent_name: str):
    return self.ollama_models["AGENT_MODELS"][agent_name]
  
  def get_system_prompt(self, agent_name: str):
    key = agent_name.lower().replace(" ", "_")
    return self.system_prompts.get(key)
  
settings = Settings()