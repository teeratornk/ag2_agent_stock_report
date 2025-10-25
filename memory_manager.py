import json
import os
from datetime import datetime
from typing import Dict, Any, List

class MemoryManager:
    """Manages memory persistence and artifact storage across iterations."""
    
    def __init__(self, config):
        self.config = config
        self.memory_file = os.path.join(config.artifact_dir, "memory.json")
        self.memory = self.load_memory()
    
    def load_memory(self) -> Dict[str, Any]:
        """Load existing memory from artifact storage."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            "iterations": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_memory(self):
        """Save current memory to artifact storage."""
        self.memory["last_updated"] = datetime.now().isoformat()
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
    
    def save_iteration_summary(self, iteration: int, summary: Dict[str, Any]):
        """Save summary of an iteration to memory."""
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
        
        # Save iteration artifact
        artifact_file = os.path.join(
            self.config.artifact_dir,
            f"iteration_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(artifact_file, "w", encoding="utf-8") as f:
            json.dump(iteration_data, f, indent=2, ensure_ascii=False)
        
        # Update memory
        self.memory["iterations"].append({
            "iteration": iteration,
            "artifact_file": artifact_file,
            "summary": summary.get("summary_text", ""),
            "timestamp": iteration_data["timestamp"]
        })
        
        self.save_memory()
    
    def save_artifact_reference(self, iteration: int, filepath: str):
        """Save a reference to an artifact file for an iteration."""
        print(f"  📝 Memory Manager: Saving artifact reference for iteration {iteration}")  # Debug
        
        # Ensure memory iterations list exists
        if "iterations" not in self.memory:
            self.memory["iterations"] = []
        
        # Update or add the iteration entry
        found = False
        for item in self.memory["iterations"]:
            if item.get("iteration") == iteration:
                item["artifact_file"] = filepath
                found = True
                print(f"    - Updated existing entry for iteration {iteration}")
                break
        
        if not found:
            # Add new entry
            self.memory["iterations"].append({
                "iteration": iteration,
                "artifact_file": filepath,
                "timestamp": datetime.now().isoformat()
            })
            print(f"    - Added new entry for iteration {iteration}")
        
        # Save the updated memory
        self.save_memory()
        print(f"    - Memory saved to {self.memory_file}")
    
    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """Get history of all iterations."""
        return self.memory.get("iterations", [])
    
    def get_last_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of the last iteration."""
        iterations = self.memory.get("iterations", [])
        if iterations:
            return iterations[-1]
        return {}
    
    def clear_memory(self):
        """Clear all memory and artifacts."""
        self.memory = {
            "iterations": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        self.save_memory()
        
        # Clean artifact files
        for file in os.listdir(self.config.artifact_dir):
            if file.startswith("iteration_") and file.endswith(".json"):
                os.remove(os.path.join(self.config.artifact_dir, file))
