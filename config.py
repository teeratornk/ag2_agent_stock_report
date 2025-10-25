import os
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Azure OpenAI and system settings."""
    
    def __init__(self):
        """Initialize configuration and create necessary directories."""
        # Set base directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up directory structure with FULL PATHS
        self.output_dir = os.path.join(self.base_dir, "output")
        self.coding_dir = os.path.join(self.base_dir, "coding")
        self.draft_dir = os.path.join(self.base_dir, "draft")
        self.artifact_dir = os.path.join(self.base_dir, "artifact")
        # NOTE: memory_dir removed - we use artifact_dir for all persistence
        
        # NOTE: Directory creation is handled by main.py setup_directories()
        # This avoids creating directories during import and gives us control
        # over when and how directories are created/cleared
        
        # Load environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        
        # Default model - this is the primary fallback
        self.default_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4").strip()
        
        # Role specific models with fallback to default_model
        self.model_writer = os.getenv("AZURE_OPENAI_CODE_WRITER", self.default_model).strip()
        self.model_critic = os.getenv("AZURE_OPENAI_CODE_CRITIC", self.default_model).strip()
        self.model_exe = os.getenv("AZURE_OPENAI_CODE_EXE", self.default_model).strip()
        
        # Working root and subdirs for coding directory
        self.working_root = self.coding_dir  # This is now the full path
        self.subdirs = {
            "data": "data",           # Just "data", not "coding/data"
            "figures": "figures",     # Just "figures", not "coding/figures"
        }
        # Sanitize in case environment or previous code injected full paths
        self.subdirs = {k: self._sanitize_subdir(v) for k, v in self.subdirs.items()}

    def _sanitize_subdir(self, value: str) -> str:
        """Return only the leaf folder name; strip any leading working_root or path components."""
        value = value.strip().replace("\\", "/")
        if value.startswith(self.working_root + "/"):
            value = value[len(self.working_root) + 1:]
        value = value.lstrip("/.")
        parts = [p for p in value.split("/") if p and p not in (".", "..")]
        return parts[-1] if parts else value

    def validate(self) -> tuple[bool, str]:
        missing = [k for k, v in {
            "AZURE_OPENAI_API_KEY": self.api_key,
            "AZURE_OPENAI_API_VERSION": self.api_version,
            "AZURE_OPENAI_ENDPOINT": self.endpoint,
        }.items() if not v]
        if missing:
            return False, f"Missing environment variables: {', '.join(missing)}"
        return True, "Configuration valid"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for autogen agents."""
        return {
            "config_list": [
                {
                    "model": self.default_model,
                    "api_key": self.api_key,
                    "base_url": self.endpoint,  # Use endpoint directly as it's already a full URL
                    "api_type": "azure",
                    "api_version": self.api_version,
                }
            ],
            "timeout": 120,
            "seed": 42,
        }

    def ensure_working_subdirs(self):
        """Create standard working subdirectories (idempotent)."""
        for subdir_name in self.subdirs.values():
            leaf = self._sanitize_subdir(subdir_name)
            # Use the full path working_root which is already a full path
            path = os.path.join(self.working_root, leaf)
            os.makedirs(path, exist_ok=True)

    def path_in_working(self, *parts: str) -> str:
        """Return a path guaranteed to be under working_root."""
        # working_root is now a full path
        return os.path.join(self.working_root, *parts)
    
    def get_subdir_path(self, key: str) -> str:
        """Get the full path for a subdirectory by its key."""
        if key in self.subdirs:
            # working_root is now a full path
            return os.path.join(self.working_root, self.subdirs[key])
        return None

_env = Config()
# DON'T call ensure_working_subdirs() here - let main.py handle it
# _env.ensure_working_subdirs()  # REMOVED - this was causing early directory creation
llm_config = _env.get_llm_config()
