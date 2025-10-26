import json
import os
from datetime import datetime
from typing import Dict, Any, List

class MemoryManager:
    """Manages memory persistence through artifact references."""
    
    def __init__(self, config):
        self.config = config
        # We don't save a memory.json file anymore - everything is in artifacts
        self.memory = {"iterations": []}
    
    def load_memory(self) -> Dict[str, Any]:
        """Load memory by scanning artifact directory."""
        # Instead of loading from memory.json, scan artifact directory
        self.memory = {
            "iterations": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Scan artifact directory for existing iteration summaries
        if os.path.exists(self.config.artifact_dir):
            for filename in os.listdir(self.config.artifact_dir):
                if filename.startswith("summary_outer") and filename.endswith(".json"):
                    # Extract iteration number from filename
                    try:
                        parts = filename.replace("summary_outer", "").replace(".json", "").split("_")
                        if parts and parts[0].isdigit():
                            iteration_num = int(parts[0])
                            self.memory["iterations"].append({
                                "iteration": iteration_num,
                                "artifact_file": os.path.join(self.config.artifact_dir, filename),
                                "timestamp": datetime.now().isoformat()
                            })
                    except:
                        continue
        
        return self.memory
    
    def save_memory(self):
        """No-op since we don't save memory.json anymore."""
        # Everything is in the artifact folder
        pass
    
    def save_iteration_summary(self, iteration: int, summary: Dict[str, Any]):
        """No-op - summaries are saved directly to artifact folder."""
        # This is now handled by artifact_manager.save_iteration_summary_to_artifact
        pass
    
    def save_artifact_reference(self, iteration: int, filepath: str):
        """No-op - we don't track references in memory.json anymore."""
        # Everything is self-contained in the artifact folder
        pass
    
    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """Get history by scanning artifact directory."""
        # Reload to get fresh data
        self.load_memory()
        return self.memory.get("iterations", [])
    
    def get_last_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of the last iteration from artifacts."""
        history = self.get_iteration_history()
        if history:
            # Sort by iteration number
            history.sort(key=lambda x: x.get("iteration", 0))
            return history[-1]
        return {}
    
    def clear_memory(self):
        """Clear memory (no-op since we don't use memory.json)."""
        self.memory = {
            "iterations": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
