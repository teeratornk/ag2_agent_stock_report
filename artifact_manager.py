import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class ArtifactManager:
    """Manages all artifact-related operations including code, drafts, and summaries."""
    
    def __init__(self, config, memory_manager):
        self.config = config
        self.memory_manager = memory_manager
        
        # Track lineage
        self.code_lineage: List[Dict[str, Any]] = []
        self.draft_lineage: List[Dict[str, Any]] = []
        
        # Current iteration tracking (will be updated by group chat manager)
        self.current_outer_turn = 0
        self.current_inner_turn = 0
    
    def set_iteration(self, outer_turn: int, inner_turn: int):
        """Update current iteration numbers."""
        print(f"    🔧 ArtifactManager.set_iteration() called:")
        print(f"       - Received parameters: outer={outer_turn}, inner={inner_turn}")
        print(f"       - Current values BEFORE: outer={self.current_outer_turn}, inner={self.current_inner_turn}")
        
        # Update the values
        self.current_outer_turn = outer_turn
        self.current_inner_turn = inner_turn
        
        print(f"       - Current values AFTER: outer={self.current_outer_turn}, inner={self.current_inner_turn}")
        
        # Verify the assignment worked
        if self.current_outer_turn != outer_turn:
            print(f"       ❌ ERROR: Assignment failed! Expected outer={outer_turn}, got {self.current_outer_turn}")
        if self.current_inner_turn != inner_turn:
            print(f"       ❌ ERROR: Assignment failed! Expected inner={inner_turn}, got {self.current_inner_turn}")
    
    # ========== CODE ARTIFACTS ==========
    
    def save_code_iteration(self, code: str, success: bool = False, feedback: str = "") -> str:
        """Save code iteration with execution status and feedback."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"code_v{self.current_outer_turn}_{self.current_inner_turn}_{timestamp}.py"
        filepath = os.path.join(self.config.coding_dir, filename)
        
        # Create header with metadata
        header = self._create_code_header(timestamp, success, feedback)
        
        # Save code with metadata
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(code)
        
        # Track in lineage
        entry = {
            "iteration": f"{self.current_outer_turn}_{self.current_inner_turn}",
            "file": filepath,
            "success": success,
            "feedback": feedback,
            "timestamp": timestamp
        }
        self.code_lineage.append(entry)
        
        print(f"  📝 Saved code iteration to {filename} (Status: {'✓' if success else '✗'})")
        return filepath
    
    def _create_code_header(self, timestamp: str, success: bool, feedback: str) -> str:
        """Create standardized header for code files."""
        execution_status = "PENDING"
        if success is True:
            execution_status = "SUCCESS"
        elif success is False:
            execution_status = "FAILED"
            
        return f'''"""
Iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
Timestamp: {timestamp}
Execution Status: {execution_status}
Feedback from Planner:
{feedback if feedback else 'No feedback yet'}
"""

'''
    
    def update_code_execution_status(self, success: bool):
        """Update the execution status of the most recent code entry."""
        if self.code_lineage:
            self.code_lineage[-1]["success"] = success
            self._rewrite_code_header(self.code_lineage[-1])
            print(f"    ✅ Updated code status: {'SUCCESS' if success else 'FAILED'}")
    
    def add_feedback_to_code(self, feedback: str):
        """Add feedback to the most recent code entry."""
        if self.code_lineage and not self.code_lineage[-1].get("feedback"):
            self.code_lineage[-1]["feedback"] = feedback
            self._rewrite_code_header(self.code_lineage[-1])
            print(f"    ✅ Added feedback to code")
    
    def _rewrite_code_header(self, entry: Dict[str, Any]):
        """Rewrite the header of a saved code file with updated feedback and status."""
        path = entry["file"]
        if not os.path.exists(path):
            return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find the end of the header docstring
        start_quotes = content.find('"""')
        if start_quotes == -1:
            return
        
        end_quotes = content.find('"""', start_quotes + 3)
        if end_quotes == -1:
            return
        
        # Get the code part after the header
        code_part = content[end_quotes + 3:].lstrip('\n')
        
        # Create new header
        header = self._create_code_header(
            entry.get("timestamp", "unknown"),
            entry.get("success", False),
            entry.get("feedback", "")
        )
        
        # Write updated file
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(code_part)
    
    def get_latest_code_with_feedback(self) -> str:
        """Get the latest code with planner's feedback for engineer to iterate."""
        if not self.code_lineage:
            return ""
        
        latest = self.code_lineage[-1]
        
        # Read the file
        if not os.path.exists(latest["file"]):
            return ""
        
        with open(latest["file"], "r", encoding="utf-8") as f:
            code = f.read()
        
        # Format for engineer with clear feedback
        formatted = f'''# Previous Code Iteration
# Status: {'SUCCESS' if latest.get("success") else 'FAILED'}
# Planner's Feedback:
# {latest.get("feedback", "No feedback yet")}
# 
# Please address the feedback above in your next iteration.
# Original code below:
# {'='*60}

{code}
'''
        return formatted
    
    # ========== DRAFT ARTIFACTS ==========
    
    def save_draft_iteration(self, draft: str, feedback: str = "") -> str:
        """Save draft iteration with feedback."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_v{self.current_outer_turn}_{self.current_inner_turn}_{timestamp}.md"
        filepath = os.path.join(self.config.draft_dir, filename)
        
        # Create header with metadata
        header = self._create_draft_header(timestamp, feedback)
        
        # Save draft with metadata
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(draft)
        
        # Track in lineage
        entry = {
            "iteration": f"{self.current_outer_turn}_{self.current_inner_turn}",
            "file": filepath,
            "feedback": feedback,
            "timestamp": timestamp
        }
        self.draft_lineage.append(entry)
        
        print(f"  📄 Saved draft iteration to {filename}")
        return filepath
    
    def _create_draft_header(self, timestamp: str, feedback: str) -> str:
        """Create standardized header for draft files."""
        return f'''---
iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
timestamp: {timestamp}
feedback: |
  {feedback if feedback else 'No feedback yet'}
---

'''
    
    def add_feedback_to_draft(self, feedback: str):
        """Add feedback to the most recent draft entry."""
        if self.draft_lineage and not self.draft_lineage[-1].get("feedback"):
            self.draft_lineage[-1]["feedback"] = feedback
            self._rewrite_draft_header(self.draft_lineage[-1])
            print(f"    ✅ Added feedback to draft")
    
    def _rewrite_draft_header(self, entry: Dict[str, Any]):
        """Rewrite the header of a saved draft file with updated feedback."""
        path = entry["file"]
        if not os.path.exists(path):
            return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract the body (everything after the front matter)
        body = content
        if content.startswith("---"):
            second = content.find("---", 3)
            if second != -1:
                body = content[second + 3:].lstrip()
        
        # Create new header
        header = self._create_draft_header(
            entry.get("timestamp", "unknown"),
            entry.get("feedback", "")
        )
        
        # Write updated file
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(body)
    
    def get_latest_draft_with_feedback(self) -> str:
        """Get the latest draft with planner's feedback for writer to iterate."""
        if not self.draft_lineage:
            return ""
        
        latest = self.draft_lineage[-1]
        
        # Read the file
        if not os.path.exists(latest["file"]):
            return ""
        
        with open(latest["file"], "r", encoding="utf-8") as f:
            draft = f.read()
        
        # Format for writer with clear feedback
        formatted = f'''<!-- Previous Draft Iteration -->
<!-- Planner's Feedback:
{latest.get("feedback", "No feedback yet")}

Please address the feedback above in your next iteration.
Original draft below:
{'='*60}
-->

{draft}
'''
        return formatted
    
    # ========== ITERATION SUMMARIES ==========
    
    def save_iteration_summary_to_artifact(self, summary: Dict[str, Any]) -> str:
        """Save iteration summary to artifact folder for persistence."""
        # Ensure artifact directory exists
        if not os.path.exists(self.config.artifact_dir):
            os.makedirs(self.config.artifact_dir, exist_ok=True)
            print(f"  📁 Created artifact directory: {self.config.artifact_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # DEBUG: Print the current iteration values
        print(f"  🔍 DEBUG in save_iteration_summary_to_artifact:")
        print(f"     self.current_outer_turn = {self.current_outer_turn}")
        print(f"     summary.get('iteration') = {summary.get('iteration', 'NOT SET')}")
        
        # USE the iteration from the summary if it's different (it should be authoritative)
        actual_iteration = summary.get("outer_iteration", self.current_outer_turn)
        if actual_iteration != self.current_outer_turn:
            print(f"  ⚠️ WARNING: Summary iteration ({actual_iteration}) differs from current ({self.current_outer_turn})")
            print(f"     Using summary iteration: {actual_iteration}")
        
        filename = f"summary_outer{actual_iteration}_{timestamp}.json"
        filepath = os.path.join(self.config.artifact_dir, filename)
        
        print(f"  💾 Saving artifact: {filename}")
        
        # Prepare the complete summary with all lineage information
        complete_summary = {
            "outer_iteration": self.current_outer_turn,
            "total_inner_iterations": self.current_inner_turn,
            "timestamp": timestamp,
            "summary_text": summary.get("summary_text", ""),
            "metrics": summary.get("metrics", {}),
            "code_lineage": summary.get("code_lineage", self.code_lineage[-5:] if self.code_lineage else []),
            "draft_lineage": summary.get("draft_lineage", self.draft_lineage[-5:] if self.draft_lineage else []),
            "artifacts": summary.get("artifacts", {}),
            "task_history": summary.get("task_history", [])
        }
        
        # Write to file with error handling
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(complete_summary, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Successfully saved iteration summary to {filename}")
            
            # Verify it's readable
            with open(filepath, "r", encoding="utf-8") as f:
                test_load = json.load(f)
                print(f"  ✅ Verified: Artifact is readable (contains {len(test_load)} keys)")
            
        except Exception as e:
            print(f"  ❌ ERROR saving artifact: {e}")
            import traceback
            traceback.print_exc()
            return filepath
        
        # Register with memory manager
        if self.memory_manager:
            try:
                self.memory_manager.save_artifact_reference(self.current_outer_turn, filepath)
                print(f"  📝 Registered artifact with memory manager for iteration {self.current_outer_turn}")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not register with memory manager: {e}")
        
        return filepath
    
    def save_planner_message(self, content: str, message_count: int) -> str:
        """Save Planner message to a file for tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"planner_msg_{self.current_outer_turn}_{message_count}_{timestamp}.txt"
        filepath = os.path.join(self.config.coding_dir, filename)
        
        # Save the planner message with context
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"=== PLANNER MESSAGE ===\n")
            f.write(f"Outer Iteration: {self.current_outer_turn}\n")
            f.write(f"Message Number: {message_count}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            f.write(content)
            f.write(f"\n\n{'='*50}\n")
            f.write("END OF MESSAGE\n")
        
        print(f"    ✅ Saved Planner feedback to {filename}")
        return filepath
    
    def save_summarizer_output(self, content: str) -> str:
        """Save Summarizer output to artifact directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # DEBUG: Print current iteration
        print(f"  🔍 DEBUG in save_summarizer_output: current_outer_turn = {self.current_outer_turn}")
        
        filename = f"summarizer_output_{self.current_outer_turn}_{timestamp}.txt"
        filepath = os.path.join(self.config.artifact_dir, filename)
        
        # Save the summarizer message with context
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"=== SUMMARIZER OUTPUT ===\n")
            f.write(f"Outer Iteration: {self.current_outer_turn}\n")
            f.write(f"Inner Turn: {self.current_inner_turn}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            f.write(content)
            f.write(f"\n\n{'='*50}\n")
            f.write("END OF SUMMARY\n")
        
        print(f"    ✅ Saved Summarizer output to {filename}")
        return filepath
    
    # ========== LINEAGE ACCESSORS ==========
    
    def get_code_lineage(self) -> List[Dict[str, Any]]:
        """Get the complete code lineage."""
        return self.code_lineage
    
    def get_draft_lineage(self) -> List[Dict[str, Any]]:
        """Get the complete draft lineage."""
        return self.draft_lineage
    
    def get_latest_code_entry(self) -> Optional[Dict[str, Any]]:
        """Get the most recent code entry."""
        return self.code_lineage[-1] if self.code_lineage else None
    
    def get_latest_draft_entry(self) -> Optional[Dict[str, Any]]:
        """Get the most recent draft entry."""
        return self.draft_lineage[-1] if self.draft_lineage else None
    
    def clear_lineages(self):
        """Clear all lineages (useful for new outer iterations)."""
        # Note: We typically don't want to clear lineages completely
        # as we want to track progress across outer iterations
        pass
    
    def get_iteration_stats(self) -> Dict[str, Any]:
        """Get statistics about the current iteration."""
        return {
            "outer_iteration": self.current_outer_turn,
            "inner_iteration": self.current_inner_turn,
            "codes_created": len(self.code_lineage),
            "drafts_created": len(self.draft_lineage),
            "last_code_success": self.code_lineage[-1]["success"] if self.code_lineage else None,
            "has_feedback": bool(
                (self.code_lineage and self.code_lineage[-1].get("feedback")) or
                (self.draft_lineage and self.draft_lineage[-1].get("feedback"))
            )
        }
