import autogen
from typing import Dict, List, Any, Optional
import json, os, hashlib, time
from datetime import datetime
from iteration_manager import IterationManager  # Add this import

class CustomGroupChatManagerWithTracking(autogen.GroupChatManager):
    """Custom GroupChatManager that tracks and saves artifacts as messages are processed."""
    
    def __init__(self, *args, custom_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_manager = custom_manager  # Reference to our CustomGroupChatManager
        self.planner_message_count = 0  # Track number of planner messages
        self.message_count = 0  # Track total messages for inner turn calculation
        
    def _process_received_message(self, message, sender, silent=False):
        """Process messages and save artifacts in real-time."""
        # Call parent implementation first
        result = super()._process_received_message(message, sender, silent)
        
        # Increment message count for agents (not Admin)
        sender_name = sender.name if hasattr(sender, 'name') else str(sender)
        if sender_name in ["Engineer", "Writer", "Executor", "Planner", "Summarizer"]:
            self.message_count += 1
            # Update inner turn count in custom_manager
            if self.custom_manager:
                self.custom_manager.current_inner_turn = max(1, (self.message_count + 3) // 4)  # Roughly 1 turn per 4 messages
        
        # Extract message content and sender name
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        sender_name = sender.name if hasattr(sender, 'name') else str(sender)
        
        print(f"  📨 Processing message from {sender_name} (length: {len(content)})")  # Debug
        
        # Save artifacts based on sender
        if self.custom_manager:
            try:
                # Save code when Engineer produces it
                if sender_name == "Engineer" and "```python" in content:
                    print(f"    🔧 Engineer code detected")  # Debug
                    code_blocks = self.custom_manager._extract_python_blocks(content)
                    for code in code_blocks:
                        if code.strip():
                            # Determine success by checking if there's an Executor message coming
                            success = False  # Will be updated later by Executor
                            filepath = self.custom_manager.save_code_iteration(code, success)
                            print(f"    ✅ Saved code to {filepath}")
                
                # Save draft when Writer produces it
                elif sender_name == "Writer" and ("```md" in content or "```markdown" in content or "#" in content):
                    print(f"    ✏️ Writer draft detected")  # Debug
                    # Try to extract markdown blocks first
                    draft_blocks = self.custom_manager._extract_md_blocks(content)
                    
                    # If no blocks found but content looks like markdown, use the whole content
                    if not draft_blocks and "#" in content and len(content) > 150:
                        draft_blocks = [content]
                    
                    for draft in draft_blocks:
                        if draft.strip():
                            filepath = self.custom_manager.save_draft_iteration(draft)
                            print(f"    ✅ Saved draft to {filepath}")
                
                # Update code execution status when Executor reports
                elif sender_name == "Executor":
                    print(f"    ⚙️ Executor results detected")  # Debug
                    # Check if execution was successful
                    success = not any(err in content.lower() for err in ["error", "exception", "traceback", "failed"])
                    
                    # Update the last code entry's success status
                    if self.custom_manager.code_lineage:
                        self.custom_manager.code_lineage[-1]["success"] = success
                        # Rewrite the file header with updated status
                        self.custom_manager._rewrite_code_header(self.custom_manager.code_lineage[-1])
                        print(f"    ✅ Updated code status: {'SUCCESS' if success else 'FAILED'}")
                
                # Capture feedback from Planner - SAVE IT TO FILE
                elif sender_name == "Planner":
                    print(f"    📋 Planner feedback detected")  # Debug
                    self.planner_message_count += 1
                    
                    # Save Planner message to a file in coding directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    planner_filename = f"planner_msg_{self.custom_manager.current_outer_turn}_{self.planner_message_count}_{timestamp}.txt"
                    planner_filepath = os.path.join(self.custom_manager.config.coding_dir, planner_filename)
                    
                    # Save the planner message with context
                    with open(planner_filepath, "w", encoding="utf-8") as f:
                        f.write(f"=== PLANNER MESSAGE ===\n")
                        f.write(f"Outer Iteration: {self.custom_manager.current_outer_turn}\n")
                        f.write(f"Message Number: {self.planner_message_count}\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(content)
                        f.write(f"\n\n{'='*50}\n")
                        f.write("END OF MESSAGE\n")
                    
                    print(f"    ✅ Saved Planner feedback to {planner_filename}")
                    
                    # Check if this is feedback for code or draft
                    # Look at recent messages to determine context
                    if len(self.groupchat.messages) > 1:
                        prev_msg = self.groupchat.messages[-2] if len(self.groupchat.messages) > 1 else None
                        if prev_msg:
                            prev_sender = prev_msg.get("name", "")
                            if prev_sender == "Executor" and self.custom_manager.code_lineage:
                                # This is feedback for code
                                self.custom_manager.code_lineage[-1]["feedback"] = content
                                self.custom_manager._rewrite_code_header(self.custom_manager.code_lineage[-1])
                                print(f"    ✅ Added feedback to code")
                            elif prev_sender == "Writer" and self.custom_manager.draft_lineage:
                                # This is feedback for draft
                                self.custom_manager.draft_lineage[-1]["feedback"] = content
                                self.custom_manager._rewrite_draft_header(self.custom_manager.draft_lineage[-1])
                                print(f"    ✅ Added feedback to draft")
                
                # Capture summary from Summarizer
                elif sender_name == "Summarizer":
                    print(f"    📊 Summarizer output detected")  # Debug
                    
                    # Save Summarizer output to a file immediately
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    summary_filename = f"summarizer_output_{self.custom_manager.current_outer_turn}_{timestamp}.txt"
                    summary_filepath = os.path.join(self.custom_manager.config.artifact_dir, summary_filename)
                    
                    # Save the summarizer message with context
                    with open(summary_filepath, "w", encoding="utf-8") as f:
                        f.write(f"=== SUMMARIZER OUTPUT ===\n")
                        f.write(f"Outer Iteration: {self.custom_manager.current_outer_turn}\n")
                        f.write(f"Inner Turn: {self.custom_manager.current_inner_turn}\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(content)
                        f.write(f"\n\n{'='*50}\n")
                        f.write("END OF SUMMARY\n")
                    
                    print(f"    ✅ Saved Summarizer output to {summary_filename}")
                    
                    # Store reference for later use in summarize_iteration
                    self.custom_manager.last_summarizer_output = content
            
            except Exception as e:
                print(f"    ❌ Error saving artifact: {e}")  # Debug
        
        return result


class CustomGroupChatManager:
    """Custom group chat manager with inner/outer iteration control."""
    
    def __init__(self, agents: Dict[str, autogen.ConversableAgent], 
                 config, memory_manager, 
                 max_inner_turn: int = 10, 
                 max_outer_turn: int = 3,
                 exit_terms: set = None):
        self.agents = agents
        self.config = config
        self.memory_manager = memory_manager
        self.max_inner_turn = max_inner_turn
        self.max_outer_turn = max_outer_turn
        self.current_outer_turn = 0
        self.current_inner_turn = 0
        
        # Initialize iteration manager
        self.iteration_manager = IterationManager(config, memory_manager)
        
        # Track lineage
        self.code_lineage = []  # List of (code, success, feedback) tuples
        self.draft_lineage = []  # List of (draft, feedback) tuples
        
        # Track explicit user termination - SINGLE SOURCE OF TRUTH for exit terms
        self.conversation_terminated = False
        if exit_terms:
            self._exit_terms = exit_terms
        else:
            self._exit_terms = {"exit", "terminate", "stop", "end conversation", "quit", "approved", "bye", "goodbye"}
        
        # Setup group chat with speaker transitions
        self.setup_group_chat()
        
        # Hash sets to prevent duplicate saves
        self._saved_code_hashes: set[str] = set()
        self._saved_draft_hashes: set[str] = set()
        
        # Track message processing
        self.last_processed_message_index = 0
        
        # Counter for unique filenames
        self.file_counter = 0
        
        # Track last summarizer output
        self.last_summarizer_output = None
        
    def setup_group_chat(self):
        """Configure group chat with allowed speaker transitions."""
        # Define speaker transitions
        allowed_transitions = {
            self.agents["user_proxy"]: [self.agents["planner"]],
            self.agents["planner"]: [
                self.agents["summarizer"], 
                self.agents["engineer"], 
                self.agents["writer"]
            ],
            self.agents["engineer"]: [self.agents["executor"]],
            self.agents["executor"]: [self.agents["planner"]],
            self.agents["writer"]: [self.agents["planner"]],
            self.agents["summarizer"]: [self.agents["user_proxy"]],
        }
        
        # Define termination function using class exit terms
        def is_termination_msg(msg):
            """Check if the message indicates termination."""
            if isinstance(msg, dict):
                content = msg.get("content", "").lower()
                name = msg.get("name", "")
                # Check for termination signals using class-level exit terms
                if name == "Admin" and any(term in content for term in self._exit_terms):
                    return True
                # Also check for the termination run messages
                if "terminating run" in content.lower():
                    return True
            return False
        
        # Create group chat with termination handling
        # max_round is our max_inner_turn - AutoGen handles the iteration internally
        self.groupchat = autogen.GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=self.max_inner_turn,  # This controls the inner iterations
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
        )
        
        # Create our custom group chat manager with tracking
        self.manager = CustomGroupChatManagerWithTracking(
            groupchat=self.groupchat, 
            llm_config=self.config.get_llm_config(),
            is_termination_msg=is_termination_msg,  # Add termination check
            custom_manager=self  # Pass reference to self for artifact saving
        )
    
    def save_code_iteration(self, code: str, success: bool, feedback: str = "") -> str:
        """Save code iteration with execution status and feedback."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"code_v{self.current_outer_turn}_{self.current_inner_turn}_{timestamp}.py"
        filepath = os.path.join(self.config.coding_dir, filename)
        
        # Create header with metadata
        header = f'''"""
Iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
Timestamp: {timestamp}
Execution Status: {'SUCCESS' if success else 'FAILED'}
Feedback from Planner:
{feedback if feedback else 'No feedback yet'}
"""

'''
        
        # Save code with metadata
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(code)
        
        # Track in lineage
        self.code_lineage.append({
            "iteration": f"{self.current_outer_turn}_{self.current_inner_turn}",
            "file": filepath,
            "success": success,
            "feedback": feedback,
            "timestamp": timestamp
        })
        
        print(f"  📝 Saved code iteration to {filename} (Status: {'✓' if success else '✗'})")
        return filepath
    
    def save_draft_iteration(self, draft: str, feedback: str = "") -> str:
        """Save draft iteration with feedback."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_v{self.current_outer_turn}_{self.current_inner_turn}_{timestamp}.md"
        filepath = os.path.join(self.config.draft_dir, filename)
        
        # Create header with metadata
        header = f'''---
iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
timestamp: {timestamp}
feedback: |
  {feedback if feedback else 'No feedback yet'}
---

'''
        
        # Save draft with metadata
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(draft)
        
        # Track in lineage
        self.draft_lineage.append({
            "iteration": f"{self.current_outer_turn}_{self.current_inner_turn}",
            "file": filepath,
            "feedback": feedback,
            "timestamp": timestamp
        })
        
        print(f"  📄 Saved draft iteration to {filename}")
        return filepath
    
    def get_latest_code_with_feedback(self) -> str:
        """Get the latest code with planner's feedback for engineer to iterate."""
        if not self.code_lineage:
            return ""
        
        latest = self.code_lineage[-1]
        with open(latest["file"], "r", encoding="utf-8") as f:
            code = f.read()
        
        # Format for engineer with clear feedback
        formatted = f'''# Previous Code Iteration
# Status: {'SUCCESS' if latest["success"] else 'FAILED'}
# Planner's Feedback:
# {latest["feedback"]}
# 
# Please address the feedback above in your next iteration.
# Original code below:
# {'='*60}

{code}
'''
        return formatted
    
    def get_latest_draft_with_feedback(self) -> str:
        """Get the latest draft with planner's feedback for writer to iterate."""
        if not self.draft_lineage:
            return ""
        
        latest = self.draft_lineage[-1]
        with open(latest["file"], "r", encoding="utf-8") as f:
            draft = f.read()
        
        # Format for writer with clear feedback
        formatted = f'''<!-- Previous Draft Iteration -->
<!-- Planner's Feedback:
{latest["feedback"]}

Please address the feedback above in your next iteration.
Original draft below:
{'='*60}
-->

{draft}
'''
        return formatted
    
    def save_iteration_summary_to_artifact(self, summary: Dict[str, Any]):
        """Save iteration summary to artifact folder for persistence."""
        # Ensure artifact directory exists (config.artifact_dir is now a full path)
        if not os.path.exists(self.config.artifact_dir):
            os.makedirs(self.config.artifact_dir, exist_ok=True)
            print(f"  📁 Created artifact directory: {self.config.artifact_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_outer{self.current_outer_turn}_{timestamp}.json"
        filepath = os.path.join(self.config.artifact_dir, filename)
        
        print(f"  💾 Saving artifact: {filename}")  # More concise debug
        
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
            "task_history": getattr(self, "task_history", [])
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
            return filepath  # Return path even if there was an error
        
        # Also save to memory manager with the filepath
        if hasattr(self, 'memory_manager') and self.memory_manager:
            try:
                self.memory_manager.save_artifact_reference(self.current_outer_turn, filepath)
                print(f"  📝 Registered artifact with memory manager for iteration {self.current_outer_turn}")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not register with memory manager: {e}")
        
        return filepath
    
    def load_previous_iteration_context(self) -> Dict[str, Any]:
        """Load context from previous iteration's artifact."""
        # Calculate the target iteration we're looking for
        target_iteration = self.current_outer_turn - 1
        
        print(f"  🔍 Looking for artifact from iteration {target_iteration}")  # Debug
        
        # First try to get from memory manager
        history = self.memory_manager.get_iteration_history()
        
        # If we have history, get the most recent artifact
        if history:
            print(f"  📚 Found {len(history)} items in history")  # Debug
            for item in history:
                if item.get("iteration") == target_iteration and "artifact_file" in item:
                    artifact_path = item["artifact_file"]
                    print(f"  📂 Found artifact reference: {artifact_path}")  # Debug
                    if os.path.exists(artifact_path):
                        with open(artifact_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            print(f"  ✅ Loaded artifact from memory manager reference")
                            return data
                    else:
                        print(f"  ⚠️ Artifact file not found at: {artifact_path}")
        
        # Fallback: look for the most recent artifact file directly
        if os.path.exists(self.config.artifact_dir):
            print(f"  🔍 Fallback: searching in {self.config.artifact_dir}")  # Debug
            
            # Look for files matching the pattern for the target iteration
            artifact_files = []
            for filename in os.listdir(self.config.artifact_dir):
                if filename.startswith(f"summary_outer{target_iteration}_") and filename.endswith(".json"):
                    artifact_files.append(os.path.join(self.config.artifact_dir, filename))
                    print(f"    - Found matching file: {filename}")  # Debug
            
            # Get the most recent one (sorted by filename which includes timestamp)
            if artifact_files:
                artifact_files.sort()
                most_recent = artifact_files[-1]
                print(f"  📂 Using most recent artifact: {os.path.basename(most_recent)}")  # Debug
                try:
                    with open(most_recent, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        print(f"  ✅ Loaded artifact from direct file search")
                        return data
                except Exception as e:
                    print(f"  ❌ Error loading artifact: {e}")
            else:
                print(f"  ⚠️ No artifact files found for iteration {target_iteration}")
        
        print(f"  ℹ️ No previous context found (this might be normal for iteration 1)")
        return {}
    
    def extract_code_from_messages(self) -> Optional[str]:
        """Extract latest code from conversation messages."""
        # Look for the most recent code block from Engineer
        for msg in reversed(self.groupchat.messages):
            if msg.get("name") == "Engineer":
                content = msg.get("content", "")
                # Look for Python code blocks
                if "```python" in content:
                    start = content.find("```python")
                    end = content.find("```", start + 9)
                    if start != -1 and end != -1:
                        code = content[start + 9:end].strip()
                        return code
                # Also check for code without markdown blocks
                elif "import" in content and "def " in content or "class " in content:
                    # Likely raw Python code
                    return content.strip()
        return None
    
    def extract_report_from_messages(self) -> Optional[str]:
        """Extract latest report from conversation messages."""
        # Look for the most recent markdown block from Writer
        for msg in reversed(self.groupchat.messages):
            if msg.get("name") == "Writer":
                content = msg.get("content", "")
                # Look for markdown blocks
                if "```md" in content or "```markdown" in content:
                    # Find the markdown block
                    for marker in ["```md", "```markdown"]:
                        if marker in content:
                            start = content.find(marker)
                            end = content.find("```", start + len(marker))
                            if start != -1 and end != -1:
                                report = content[start + len(marker):end].strip()
                                return report
                # Also check if the entire message is markdown content
                elif "#" in content and len(content) > 100:
                    # Likely markdown content without code blocks
                    return content.strip()
        return None
    
    def check_execution_success(self) -> bool:
        """Check if the last code execution was successful."""
        # Look for error messages in recent executor messages
        for msg in reversed(self.groupchat.messages[-10:]):
            if msg.get("name") == "Executor":
                content = msg.get("content", "").lower()
                # Check for explicit error indicators
                if any(error in content for error in ["error", "failed", "exception", "traceback"]):
                    # But make sure it's not just mentioning error handling
                    if "no error" not in content and "without error" not in content:
                        return False
                # Check for success indicators
                elif any(success in content for success in ["success", "completed", "saved", "figure saved", "data saved"]):
                    return True
        # Default to False if unsure
        return False
    
    def extract_planner_feedback_for_code(self) -> str:
        """Extract planner's feedback on the code."""
        # Look for the most recent Planner message after Executor
        executor_found = False
        for msg in reversed(self.groupchat.messages):
            if msg.get("name") == "Executor":
                executor_found = True
            elif executor_found and msg.get("name") == "Planner":
                content = msg.get("content", "")
                # Look for feedback patterns
                feedback_keywords = ["improve", "fix", "error", "suggest", "change", "modify", "update", "correct"]
                if any(keyword in content.lower() for keyword in feedback_keywords):
                    return content
                # Even if no explicit feedback keywords, if Planner comments after execution, it's feedback
                if len(content) > 50:  # Substantial message
                    return content
        return ""
    
    def extract_planner_feedback_for_draft(self) -> str:
        """Extract planner's feedback on the draft."""
        # Look for the most recent Planner message after Writer
        writer_found = False
        for msg in reversed(self.groupchat.messages):
            if msg.get("name") == "Writer":
                writer_found = True
            elif writer_found and msg.get("name") == "Planner":
                content = msg.get("content", "")
                # Look for feedback patterns related to writing
                feedback_keywords = ["revise", "improve", "add", "clarify", "seo", "update", "include", "expand"]
                if any(keyword in content.lower() for keyword in feedback_keywords):
                    return content
                # Even if no explicit feedback keywords, if Planner comments after Writer, it's feedback
                if len(content) > 50:  # Substantial message
                    return content
        return ""
    
    def execute_planning_phase(self, task: str) -> str:
        """Execute planning phase and return next state."""
        # Planner analyzes task and determines next steps
        # This would be handled by the group chat flow
        return "coding"  # Next phase after planning
    
    def execute_coding_phase(self) -> str:
        """Execute coding phase with lineage tracking."""
        # Provide previous code with feedback to engineer if available
        if self.code_lineage:
            prev_code = self.get_latest_code_with_feedback()
            # This would be sent through the group chat flow
            print(f"  📋 Providing previous code with feedback to Engineer")
        
        # Let the group chat flow naturally (Engineer writes, Executor runs)
        # This is handled by the group chat manager based on transitions
        
        # After execution, extract and save the code
        # We'll hook into the message processing to capture this
        return "writing"  # Next phase after coding
    
    def execute_writing_phase(self) -> str:
        """Execute writing phase with lineage tracking."""
        # Provide previous draft with feedback to writer if available  
        if self.draft_lineage:
            prev_draft = self.get_latest_draft_with_feedback()
            # This would be sent through the group chat flow
            print(f"  📋 Providing previous draft with feedback to Writer")
        
        # Let the group chat flow naturally (Writer creates report)
        # This is handled by the group chat manager based on transitions
        
        return "reviewing"  # Next phase after writing
    
    # ========== SCANNERS & HELPERS ==========

    def _hash(self, text: str) -> str:
        """Create a SHA-256 hash of the given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _extract_blocks(self, content: str, fence: str) -> List[str]:
        """Extract code or markdown blocks delimited by the given fence."""
        blocks = []
        start = 0
        while True:
            # Find the fence
            i = content.find(fence, start)
            if i == -1:
                break
            
            # Move past the fence and any language identifier
            fence_end = i + len(fence)
            
            # Skip any text on the same line as the opening fence (like language identifier)
            newline_after_fence = content.find('\n', fence_end)
            if newline_after_fence != -1:
                block_start = newline_after_fence + 1
            else:
                block_start = fence_end
            
            # Find the closing fence
            j = content.find("```", block_start)
            if j == -1:
                # No closing fence found, take rest of content
                block_content = content[block_start:].strip()
                if block_content:
                    blocks.append(block_content)
                break
            
            # Extract the block content
            block_content = content[block_start:j].strip()
            if block_content:  # Only add non-empty blocks
                blocks.append(block_content)
            
            # Move start position for next search
            start = j + 3
        
        return blocks

    def _extract_python_blocks(self, content: str) -> List[str]:
        """Extract Python code blocks from content."""
        return self._extract_blocks(content, "```python")

    def _extract_md_blocks(self, content: str) -> List[str]:
        """Extract Markdown blocks from content (supports ```md or ```markdown)."""
        return self._extract_blocks(content, "```md") + self._extract_blocks(content, "```markdown")

    def _determine_execution_status(self, engineer_index: int) -> bool:
        """Determine the execution status by inspecting subsequent Executor messages."""
        # Look forward for Executor message until next Engineer or Writer
        msgs = self.groupchat.messages
        for k in range(engineer_index + 1, len(msgs)):
            name = msgs[k].get("name", "")
            content = msgs[k].get("content", "").lower()
            if name in ("Engineer", "Writer"):
                break
            if name == "Executor":
                if any(x in content for x in ["error", "exception", "traceback", "failed"]):
                    return False
                if any(x in content for x in ["success", "completed", "saved", "figure saved", "data saved"]):
                    return True
        # Default: treat as failed if no explicit success
        return False

    def _attach_planner_feedback(self):
        """Attach first Planner feedback after latest unsatisfied code/draft entries."""
        if not self.groupchat.messages:
            return
        # Map last code/draft timestamp -> waiting for feedback
        last_planner_msg = None
        for msg in reversed(self.groupchat.messages):
            if msg.get("name") == "Planner":
                last_planner_msg = msg.get("content", "").strip()
                break
        if not last_planner_msg:
            return
        # If latest code entry has empty feedback -> fill
        if self.code_lineage and not self.code_lineage[-1]["feedback"]:
            self.code_lineage[-1]["feedback"] = last_planner_msg
            self._rewrite_code_header(self.code_lineage[-1])
        if self.draft_lineage and not self.draft_lineage[-1]["feedback"]:
            self.draft_lineage[-1]["feedback"] = last_planner_msg
            self._rewrite_draft_header(self.draft_lineage[-1])

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
        
        # Create new header with updated status
        execution_status = "PENDING"
        if entry.get("success") is True:
            execution_status = "SUCCESS"
        elif entry.get("success") is False:
            execution_status = "FAILED"
            
        header = f'''"""
Iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
Timestamp: {entry.get("timestamp", "unknown")}
Execution Status: {execution_status}
Feedback from Planner:
{entry.get("feedback") or "No feedback yet"}
"""

'''
        
        # Write updated file
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(code_part)

    def _rewrite_draft_header(self, entry: Dict[str, Any]):
        """Rewrite the header of a saved draft file with updated feedback."""
        path = entry["file"]
        if not os.path.exists(path): return
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        body = content
        if content.startswith("---"):
            second = content.find("---", 3)
            if second != -1:
                body = content[second+3:].lstrip()
        header = f'''---
iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
timestamp: {entry["timestamp"]}
feedback: |
  {entry["feedback"] or 'No feedback yet'}
---

'''
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + body)

    def _scan_and_save_new_code(self):
        """Scan for new code blocks in messages and save them."""
        msgs = self.groupchat.messages
        print(f"  🔍 Scanning {len(msgs)} messages for code...")  # Debug
        
        for idx, msg in enumerate(msgs):
            if msg.get("name") != "Engineer":
                continue
            
            content = msg.get("content", "") or ""
            print(f"    - Found Engineer message at index {idx}, length: {len(content)}")  # Debug
            
            blocks = self._extract_python_blocks(content)
            
            if not blocks:
                print(f"    - No Python blocks found in Engineer message")  # Debug
                continue
            
            print(f"    - Found {len(blocks)} code blocks to save")  # Debug
            
            # Save each new block
            for block in blocks:
                h = self._hash(block)
                if h in self._saved_code_hashes:
                    print(f"    - Block already saved (hash exists)")  # Debug
                    continue
                success = self._determine_execution_status(idx)
                self._saved_code_hashes.add(h)
                self._save_code_block(block, success)

    def _save_code_block(self, code: str, success: bool):
        """Save a new code block as a file and update lineage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"code_v{self.current_outer_turn}_{self.current_inner_turn}_{timestamp}.py"
        path = os.path.join(self.config.coding_dir, filename)
        header = f'''"""
Iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
Timestamp: {timestamp}
Execution Status: {'SUCCESS' if success else 'FAILED'}
Feedback from Planner:
No feedback yet
"""

'''
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + code)
        self.code_lineage.append({
            "iteration": f"{self.current_outer_turn}_{self.current_inner_turn}",
            "file": path,
            "success": success,
            "feedback": "",
            "timestamp": timestamp
        })
        print(f"  📝 Code saved: {filename} ({'✓' if success else '✗'})")

    def _scan_and_save_new_drafts(self):
        """Scan for new draft blocks in messages and save them."""
        msgs = self.groupchat.messages
        print(f"  🔍 Scanning {len(msgs)} messages for drafts...")  # Debug
        
        for idx, msg in enumerate(msgs):
            if msg.get("name") != "Writer":
                continue
            
            content = msg.get("content", "") or ""
            print(f"    - Found Writer message at index {idx}, length: {len(content)}")  # Debug
            
            # Look for markdown blocks
            blocks = self._extract_md_blocks(content)
            
            # Also check if filename directive is present for direct saving
            if "# filename: report_draft.md" in content or "filename: report_draft.md" in content:
                # Extract the content that should be saved
                if "```md" in content or "```markdown" in content:
                    # Already in blocks, will be processed
                    pass
                else:
                    # Treat the whole content after filename directive as markdown
                    lines = content.split('\n')
                    start_idx = -1
                    for i, line in enumerate(lines):
                        if "filename: report_draft.md" in line:
                            start_idx = i + 1
                            break
                    if start_idx > 0:
                        markdown_content = '\n'.join(lines[start_idx:])
                        if markdown_content.strip():
                            blocks.append(markdown_content.strip())
            
            if not blocks:
                # fallback: large markdown-like content
                if "#" in content and len(content) > 150:
                    blocks = [content]
                    print(f"    - Using fallback: found markdown-like content")  # Debug
            
            print(f"    - Found {len(blocks)} draft blocks to save")  # Debug
            
            for block in blocks:
                h = self._hash(block)
                if h in self._saved_draft_hashes:
                    print(f"    - Block already saved (hash exists)")  # Debug
                    continue
                self._saved_draft_hashes.add(h)
                self._save_draft_block(block)

    def _save_draft_block(self, draft: str):
        """Save a new draft block as a file and update lineage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_v{self.current_outer_turn}_{self.current_inner_turn}_{timestamp}.md"
        path = os.path.join(self.config.draft_dir, filename)
        header = f'''---
iteration: Outer {self.current_outer_turn}, Inner {self.current_inner_turn}
timestamp: {timestamp}
feedback: |
  No feedback yet
---

'''
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + draft)
        self.draft_lineage.append({
            "iteration": f"{self.current_outer_turn}_{self.current_inner_turn}",
            "file": path,
            "feedback": "",
            "timestamp": timestamp
        })
        print(f"  📄 Draft saved: {filename}")

    # ========== OVERRIDES / FLOW ==========

    def run(self, initial_task: str) -> Dict[str, Any]:
        """Run the complete group chat with iteration control."""
        task = initial_task
        final_result = {}
        self.conversation_terminated = False
        
        # Check for existing artifacts BEFORE starting any iterations
        artifact_summary = ""
        if self.current_outer_turn == 0:  # Only on first run
            artifact_summary = self.iteration_manager.load_and_summarize_all_artifacts()
            if artifact_summary:
                print("\n" + "=" * 80)
                print("EXISTING ARTIFACTS DETECTED")
                print("=" * 80)
                print(artifact_summary)
                print("=" * 80)
        
        while not self.conversation_terminated and self.current_outer_turn < self.max_outer_turn:
            self.current_outer_turn += 1
            self.current_inner_turn = 1  # Start at 1, not 0
            print(f"\n=== Outer Iteration {self.current_outer_turn}/{self.max_outer_turn} ===")
            
            # Clear previous messages for new outer iteration
            self.groupchat.messages.clear()
            self.last_processed_message_index = 0
            
            # Reset message counts in the tracking manager
            if hasattr(self.manager, 'planner_message_count'):
                self.manager.planner_message_count = 0
            if hasattr(self.manager, 'message_count'):
                self.manager.message_count = 0
            
            # Prepare context message
            context_message = ""
            
            # For the first iteration, include the artifact summary if it exists
            if self.current_outer_turn == 1 and artifact_summary:
                context_message = f"""
{artifact_summary}

Based on the existing artifacts found above, please analyze what has been done and determine the best approach for the current task.
"""
            
            # For subsequent iterations, load context from the previous iteration
            if self.current_outer_turn > 1:
                prev_context = self.load_previous_iteration_context()
                if prev_context:
                    # Build a context message from the previous iteration
                    context_message = self._build_context_message(prev_context)
                    
                    # Also check for any existing code and drafts from all previous iterations
                    existing_artifacts = self._get_all_previous_artifacts()
                    if existing_artifacts:
                        context_message += f"\n\n{existing_artifacts}"
            
            # Update the task to include context if we have any
            if context_message:
                task = f"""Context from previous work:
{context_message}

Current task:
{initial_task}

Please analyze what has been done and plan the next steps accordingly. If previous work exists and is satisfactory, you may focus on refinements or declare the task complete."""
            
            # Initialize conversation with user proxy for this outer iteration
            try:
                # Start the chat which will run up to max_inner_turn rounds
                print(f"--- Starting conversation (max {self.max_inner_turn} rounds) ---")
                
                # Show context being provided if any
                if context_message:
                    print("\n📚 Providing context to Planner...")
                    if artifact_summary and self.current_outer_turn == 1:
                        print(f"  - Found existing artifacts from previous sessions")
                    if self.current_outer_turn > 1:
                        print(f"  - Previous outer iterations completed: {self.current_outer_turn - 1}")
                    print(f"  - Total codes created so far: {len(self.code_lineage)}")
                    print(f"  - Total drafts created so far: {len(self.draft_lineage)}")
                
                result = self.agents["user_proxy"].initiate_chat(
                    self.manager,
                    message=task,
                    clear_history=False,  # We already cleared it above
                    silent=False
                )
            except Exception as e:
                # Check if the exception indicates termination
                error_msg = str(e).lower()
                # Check for various termination indicators in the error message
                termination_keywords = ["terminating", "no reply generated", "user requested to end", 
                                       "exit", "stop", "terminate", "quit"]
                
                if any(keyword in error_msg for keyword in termination_keywords):
                    self.conversation_terminated = True
                    print("\n🛑 Conversation terminated by user.")
                else:
                    print(f"Chat completed or interrupted: {e}")
                    # For other exceptions, we might want to continue or handle differently
                    # but for safety, we'll treat it as termination
                    self.conversation_terminated = True

            # Check for admin exit in messages (as backup)
            if not self.conversation_terminated and self._admin_exit_detected():
                self.conversation_terminated = True
                
            # If conversation was terminated, break immediately
            if self.conversation_terminated:
                print("\n🛑 Ending conversation.")
                # Process any messages that were generated before termination
                self._process_messages_and_save_artifacts()
                return {"report": "Conversation terminated by user.", "terminated": True}

            # Only continue with processing if not terminated
            # Process the messages that were generated during the chat
            inner_results = self._process_messages_and_save_artifacts()
            
            # Summarize iteration
            summary = self.summarize_iteration(inner_results)
            
            # Save to memory
            self.memory_manager.save_iteration_summary(
                self.current_outer_turn, summary
            )
            
            # Get user feedback
            user_feedback = self.get_user_feedback(summary)
            
            final_result = {
                "iteration": self.current_outer_turn,
                "inner_results": inner_results,
                "summary": summary,
                "user_feedback": user_feedback,
                "continue": user_feedback.get("continue", True)
            }
            
            if not user_feedback.get("continue", True):
                break
                
            # Update task based on feedback
            if user_feedback.get("refinements"):
                task = self.update_task_with_feedback(
                    task, user_feedback["refinements"]
                )

        # Generate final report only if we have results
        if final_result:
            final_result["report"] = self.compile_final_report(final_result)
        else:
            final_result = {"report": "Conversation terminated without generating results.", "terminated": True}
            
        return final_result
    
    def _process_messages_and_save_artifacts(self) -> Dict[str, Any]:
        """Process all messages from the chat and save artifacts."""
        results = {"codes": [], "reports": [], "completed": False}
        
        # Process any new messages - scan and save code/draft blocks
        self._process_new_messages()
        
        # Check if completion was reached
        if self.check_planner_satisfaction():
            results["completed"] = True
        
        # Update results
        results["codes"] = [c["file"] for c in self.code_lineage]
        results["reports"] = [d["file"] for d in self.draft_lineage]
        
        # Update inner turn count based on actual messages processed
        # Don't override if it's already been updated by the tracking manager
        agent_messages = [m for m in self.groupchat.messages if m.get("name") in ["Engineer", "Writer", "Executor", "Planner"]]
        if agent_messages:
            # Estimate: roughly 4 messages per inner turn (Planner->Engineer->Executor->Planner)
            estimated_turn = max(1, (len(agent_messages) + 3) // 4)
            self.current_inner_turn = max(self.current_inner_turn, estimated_turn)
        
        return results
    
    def _process_new_messages(self):
        """Process new messages and save code/draft artifacts."""
        # Scan and save any new code blocks
        self._scan_and_save_new_code()
        
        # Scan and save any new draft blocks
        self._scan_and_save_new_drafts()
        
        # Attach planner feedback to the latest code/draft if available
        self._attach_planner_feedback()
        
        # Update the last processed index
        self.last_processed_message_index = len(self.groupchat.messages)

    def run_inner_iteration(self, task: str) -> Dict[str, Any]:
        """DEPRECATED - This method is no longer used. Processing happens in _process_messages_and_save_artifacts."""
        # This method is kept for backward compatibility but should not be called
        return {"codes": [], "reports": [], "completed": False}

    def check_planner_satisfaction(self) -> bool:
        """Check if planner is satisfied with current progress."""
        # Check last messages for completion signal
        recent_messages = self.groupchat.messages[-5:] if len(self.groupchat.messages) > 5 else self.groupchat.messages
        
        for msg in recent_messages:
            if msg.get("name") == "Planner" and "REVIEW_COMPLETE" in msg.get("content", ""):
                return True
        
        return False
    
    def summarize_iteration(self, inner_results: Dict[str, Any]) -> Dict[str, Any]:
        """Use summarizer agent to create iteration summary and save to artifact."""
        print(f"\n📝 Creating iteration summary...")  # Debug
        
        # Prepare summary data
        summary_data = {
            "outer_iteration": self.current_outer_turn,
            "total_inner_iterations": self.current_inner_turn,
            "codes_created": len(self.code_lineage),
            "drafts_created": len(self.draft_lineage),
            "last_code_success": self.code_lineage[-1]["success"] if self.code_lineage else None,
            "completion_status": inner_results.get("completed", False)
        }
        
        print(f"  - Codes created: {summary_data['codes_created']}")  # Debug
        print(f"  - Drafts created: {summary_data['drafts_created']}")  # Debug
        
        # Check if we already have a summarizer output from the conversation
        if hasattr(self, 'last_summarizer_output') and self.last_summarizer_output:
            print(f"  - Using captured Summarizer output")
            summary_response = self.last_summarizer_output
        else:
            # Fallback: Generate summary if not captured during conversation
            print(f"  - Generating new summary (no captured output found)")
            
            # Trigger summarizer agent
            summary_prompt = (
                f"Summarize the progress of outer iteration {self.current_outer_turn}:\n"
                f"- Total inner iterations: {self.current_inner_turn}\n"
                f"- Codes created: {len(self.code_lineage)}\n"
                f"- Reports created: {len(self.draft_lineage)}\n"
                f"- Latest code status: {'SUCCESS' if summary_data['last_code_success'] else 'FAILED' if summary_data['last_code_success'] is not None else 'N/A'}\n"
                f"- Completion status: {inner_results.get('completed', False)}\n"
                "Provide a structured summary of key findings, improvements made, and remaining tasks."
            )
            
            # Get summary from summarizer agent
            summary_response = self.agents["summarizer"].generate_reply(
                messages=[{"content": summary_prompt, "role": "user"}]
            )
        
        # Build complete summary object with all data
        summary = {
            "iteration": self.current_outer_turn,
            "inner_turns": self.current_inner_turn,
            "artifacts": inner_results,
            "summary_text": summary_response,
            "metrics": summary_data,
            "code_lineage": self.code_lineage,  # Include full lineage
            "draft_lineage": self.draft_lineage,  # Include full lineage
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save to artifact - this is the critical part
        artifact_file = self.save_iteration_summary_to_artifact(summary)
        print(f"  ✅ Artifact file path: {artifact_file}")  # Debug
        
        # Verify the file was actually created
        if os.path.exists(artifact_file):
            print(f"  ✅ Verified: Artifact file exists on disk")
            # Check file size
            file_size = os.path.getsize(artifact_file)
            print(f"  ✅ Artifact file size: {file_size} bytes")
        else:
            print(f"  ❌ ERROR: Artifact file was not created!")
        
        # Also save to memory manager
        self.memory_manager.save_iteration_summary(self.current_outer_turn, summary)
        
        # Clear the last summarizer output for next iteration
        self.last_summarizer_output = None
        
        return summary
    
    def get_user_feedback(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Get feedback from user on current iteration."""
        print("\n--- Iteration Summary ---")
        print(summary.get("summary_text", "No summary available"))
        print("\n" + "-" * 40)
        
        # In actual implementation, this would wait for user input
        # For now, simulate feedback structure
        return {
            "continue": True,
            "refinements": "",
            "approved": False
        }
    
    def update_task_with_feedback(self, task: str, feedback: str) -> str:
        """Update task based on user feedback."""
        if feedback:
            return f"{task}\n\nAdditional requirements based on feedback:\n{feedback}"
        return task
    
    def compile_final_report(self, result: Dict[str, Any]) -> str:
        """Compile the final report from all iterations."""
        # Get the latest report
        reports = result.get("inner_results", {}).get("reports", [])
        if reports:
            latest_report = reports[-1]
            with open(latest_report, "r", encoding="utf-8") as f:
                return f.read()
        return "No report generated."
    
    def _admin_exit_detected(self) -> bool:
        """Return True if Admin signaled termination."""
        if not self.groupchat.messages:
            return False
        # Scan last few messages
        for msg in reversed(self.groupchat.messages[-8:]):
            if isinstance(msg, dict) and msg.get("name") == "Admin":
                content = (msg.get("content") or "").lower()
                if any(term in content for term in self._exit_terms) or "terminating run" in content:
                    return True
        return False

    def _build_context_message(self, prev_context: Dict[str, Any]) -> str:
        """Build a context message from previous iteration data."""
        context_parts = []
        
        # Add summary from previous iteration
        if "summary_text" in prev_context:
            context_parts.append(f"**Previous Iteration Summary:**\n{prev_context['summary_text']}")
        
        # Add information about code artifacts
        if "code_lineage" in prev_context and prev_context["code_lineage"]:
            last_code = prev_context["code_lineage"][-1]
            context_parts.append(f"\n**Last Code Status:**\n- File: {last_code.get('file', 'N/A')}\n- Success: {last_code.get('success', 'N/A')}\n- Feedback: {last_code.get('feedback', 'None')}")
        
        # Add information about draft artifacts  
        if "draft_lineage" in prev_context and prev_context["draft_lineage"]:
            last_draft = prev_context["draft_lineage"][-1]
            context_parts.append(f"\n**Last Draft Status:**\n- File: {last_draft.get('file', 'N/A')}\n- Feedback: {last_draft.get('feedback', 'None')}")
        
        # Add metrics
        if "metrics" in prev_context:
            metrics = prev_context["metrics"]
            context_parts.append(f"\n**Metrics from Previous Iteration:**\n- Codes created: {metrics.get('codes_created', 0)}\n- Drafts created: {metrics.get('drafts_created', 0)}\n- Completion status: {metrics.get('completion_status', False)}")
        
        return "\n".join(context_parts)
    
    def _get_all_previous_artifacts(self) -> str:
        """Get summary of all previous artifacts from all iterations."""
        artifact_summary = []
        
        # Get all artifact files
        artifact_files = []
        if os.path.exists(self.config.artifact_dir):
            for filename in os.listdir(self.config.artifact_dir):
                if filename.startswith("summary_outer") and filename.endswith(".json"):
                    artifact_files.append(os.path.join(self.config.artifact_dir, filename))
        
        # Sort by filename (which includes timestamp)
        artifact_files.sort()
        
        if artifact_files:
            artifact_summary.append("**All Previous Iterations Overview:**")
            for i, filepath in enumerate(artifact_files, 1):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        outer_iter = data.get("outer_iteration", i)
                        inner_turns = data.get("total_inner_iterations", 0)
                        codes = len(data.get("code_lineage", []))
                        drafts = len(data.get("draft_lineage", []))
                        artifact_summary.append(f"- Iteration {outer_iter}: {inner_turns} inner turns, {codes} codes, {drafts} drafts")
                except:
                    continue
        
        return "\n".join(artifact_summary) if artifact_summary else ""
