import autogen
from typing import Dict, List, Any, Optional
import json, os, hashlib, time
from datetime import datetime
from iteration_manager import IterationManager
from artifact_manager import ArtifactManager  # Add new import

class CustomGroupChatManagerWithTracking(autogen.GroupChatManager):
    """Custom GroupChatManager that tracks and saves artifacts as messages are processed."""
    
    def __init__(self, *args, custom_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_manager = custom_manager  # Reference to our CustomGroupChatManager
        self.planner_message_count = 0  # Track number of planner messages
        self.message_count = 0  # Track total messages for inner turn calculation
        self.admin_message_count = 0  # Track Admin messages to detect new outer iterations
        
    def _process_received_message(self, message, sender, silent=False):
        """Process messages and save artifacts in real-time."""
        # Call parent implementation first
        result = super()._process_received_message(message, sender, silent)
        
        # Extract message content and sender name
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        sender_name = sender.name if hasattr(sender, 'name') else str(sender)
        
        # CRITICAL: Detect new outer iteration when Admin sends a message after Summarizer
        if sender_name == "Admin" and self.custom_manager:
            self.admin_message_count += 1
            # Check if the previous speaker was Summarizer (indicates new outer iteration)
            if len(self.groupchat.messages) > 1:
                prev_msg = self.groupchat.messages[-2] if len(self.groupchat.messages) > 1 else None
                if prev_msg and prev_msg.get("name") == "Summarizer":
                    # This is a new outer iteration!
                    print(f"\n{'='*80}")
                    print(f"🔄 NEW OUTER ITERATION DETECTED (Admin after Summarizer)")
                    
                    # Save the current iteration's summary before incrementing
                    if self.custom_manager.last_summarizer_output:
                        print(f"  💾 Saving summary for iteration {self.custom_manager.current_outer_turn}")
                        inner_results = {
                            "completed": True,
                            "code_created": len(self.custom_manager.code_lineage) > 0,
                            "report_created": len(self.custom_manager.draft_lineage) > 0,
                            "execution_success": True,
                            "artifacts": {
                                "codes": len(self.custom_manager.code_lineage),
                                "drafts": len(self.custom_manager.draft_lineage),
                            }
                        }
                        summary = self.custom_manager.summarize_iteration(inner_results)
                    
                    # Now increment for the new iteration
                    self.custom_manager.current_outer_turn += 1
                    self.custom_manager.current_inner_turn = 1
                    self.message_count = 0  # Reset inner message count
                    
                    print(f"  ➡️ Starting Outer Iteration {self.custom_manager.current_outer_turn}")
                    print(f"{'='*80}\n")
                    
                    # Update artifact manager
                    self.custom_manager.artifact_manager.set_iteration(
                        self.custom_manager.current_outer_turn,
                        self.custom_manager.current_inner_turn
                    )
        
        # Increment message count for agents (not Admin)
        if sender_name in ["Engineer", "Writer", "Executor", "Planner", "Summarizer"]:
            self.message_count += 1
            # Update inner turn count in custom_manager
            if self.custom_manager:
                self.custom_manager.current_inner_turn = max(1, (self.message_count + 3) // 4)
                # Update artifact manager's iteration with current values
                self.custom_manager.artifact_manager.set_iteration(
                    self.custom_manager.current_outer_turn,
                    self.custom_manager.current_inner_turn
                )
        
        print(f"  📨 Processing message from {sender_name} (length: {len(content)})")
        
        # Save artifacts based on sender
        if self.custom_manager:
            try:
                # Save code when Engineer produces it
                if sender_name == "Engineer" and "```python" in content:
                    print(f"    🔧 Engineer code detected")
                    code_blocks = self.custom_manager._extract_python_blocks(content)
                    for code in code_blocks:
                        if code.strip():
                            # Save with initial status (will be updated by Executor)
                            filepath = self.custom_manager.artifact_manager.save_code_iteration(code, success=False)
                            print(f"    ✅ Saved code to {filepath}")
                
                # Save draft when Writer produces it
                elif sender_name == "Writer" and ("```md" in content or "```markdown" in content or "#" in content):
                    print(f"    ✏️ Writer draft detected")
                    draft_blocks = self.custom_manager._extract_md_blocks(content)
                    
                    if not draft_blocks and "#" in content and len(content) > 150:
                        draft_blocks = [content]
                    
                    for draft in draft_blocks:
                        if draft.strip():
                            filepath = self.custom_manager.artifact_manager.save_draft_iteration(draft)
                            print(f"    ✅ Saved draft to {filepath}")
                
                # Update code execution status when Executor reports
                elif sender_name == "Executor":
                    print(f"    ⚙️ Executor results detected")
                    success = not any(err in content.lower() for err in ["error", "exception", "traceback", "failed"])
                    self.custom_manager.artifact_manager.update_code_execution_status(success)
                
                # Capture feedback from Planner
                elif sender_name == "Planner":
                    print(f"    📋 Planner feedback detected")
                    self.planner_message_count += 1
                    
                    # Save Planner message
                    self.custom_manager.artifact_manager.save_planner_message(content, self.planner_message_count)
                    
                    # Check if this is feedback for code or draft
                    if len(self.groupchat.messages) > 1:
                        prev_msg = self.groupchat.messages[-2] if len(self.groupchat.messages) > 1 else None
                        if prev_msg:
                            prev_sender = prev_msg.get("name", "")
                            if prev_sender == "Executor":
                                self.custom_manager.artifact_manager.add_feedback_to_code(content)
                            elif prev_sender == "Writer":
                                self.custom_manager.artifact_manager.add_feedback_to_draft(content)
                
                # Capture summary from Summarizer
                elif sender_name == "Summarizer":
                    print(f"    📊 Summarizer output detected")
                    
                    # Print the summarization
                    print("\n" + "="*80)
                    print("📊 ITERATION SUMMARY")
                    print("="*80)
                    print(content)
                    print("="*80 + "\n")
                    
                    # Save the summarizer output
                    self.custom_manager.artifact_manager.save_summarizer_output(content)
                    self.custom_manager.last_summarizer_output = content
                    
                    # CRITICAL: Save the iteration summary immediately when Summarizer speaks
                    if self.custom_manager:
                        print(f"  💾 Saving iteration {self.custom_manager.current_outer_turn} summary...")
                        
                        # Create inner results based on current state
                        inner_results = {
                            "completed": True,
                            "code_created": len(self.custom_manager.code_lineage) > 0,
                            "report_created": len(self.custom_manager.draft_lineage) > 0,
                            "execution_success": self.custom_manager.check_execution_success(),
                            "artifacts": {
                                "codes": len(self.custom_manager.code_lineage),
                                "drafts": len(self.custom_manager.draft_lineage),
                            }
                        }
                        
                        # Save the JSON summary
                        summary = self.custom_manager.summarize_iteration(inner_results)
                        
                        print(f"  ✅ Iteration {self.custom_manager.current_outer_turn} summary saved")
                        
                        # Check if we've reached max outer turns
                        if self.custom_manager.current_outer_turn >= self.custom_manager.max_outer_turn:
                            print(f"\n🏁 Reached maximum outer iterations ({self.custom_manager.max_outer_turn})")
                            print("  Terminating conversation...")
                            # Signal termination by returning a special message
                            return {"content": "TERMINATE: Max iterations reached", "name": "system"}
            
            except Exception as e:
                print(f"    ❌ Error saving artifact: {e}")
        
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
        
        # Initialize managers
        self.iteration_manager = IterationManager(config, memory_manager)
        self.artifact_manager = ArtifactManager(config, memory_manager)
        
        # Track explicit user termination
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
                
                # CRITICAL: Terminate after Summarizer to move to next outer iteration
                if name == "Summarizer":
                    print("\n🔄 Summarizer completed - ending current iteration")
                    # Check if we should continue to next iteration
                    if self.current_outer_turn < self.max_outer_turn:
                        print(f"  ➡️ Will continue to iteration {self.current_outer_turn + 1}")
                    else:
                        print(f"  🏁 Reached max iterations ({self.max_outer_turn})")
                    return True  # Terminate inner loop after Summarizer
                
                # Check for termination signals using class-level exit terms
                if name == "Admin" and any(term in content for term in self._exit_terms):
                    return True
                # Also check for the termination run messages
                if "terminating run" in content.lower():
                    return True
                # Check for system termination message
                if "terminate: max iterations reached" in content.lower():
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
    
    # Delegate artifact methods to ArtifactManager
    @property
    def code_lineage(self):
        """Get code lineage from artifact manager."""
        return self.artifact_manager.get_code_lineage()
    
    @property
    def draft_lineage(self):
        """Get draft lineage from artifact manager."""
        return self.artifact_manager.get_draft_lineage()
    
    def save_code_iteration(self, code: str, success: bool, feedback: str = "") -> str:
        """Delegate to artifact manager."""
        return self.artifact_manager.save_code_iteration(code, success, feedback)
    
    def save_draft_iteration(self, draft: str, feedback: str = "") -> str:
        """Delegate to artifact manager."""
        return self.artifact_manager.save_draft_iteration(draft, feedback)
    
    def get_latest_code_with_feedback(self) -> str:
        """Delegate to artifact manager."""
        return self.artifact_manager.get_latest_code_with_feedback()
    
    def get_latest_draft_with_feedback(self) -> str:
        """Delegate to artifact manager."""
        return self.artifact_manager.get_latest_draft_with_feedback()
    
    def save_iteration_summary_to_artifact(self, summary: Dict[str, Any]) -> str:
        """Delegate to artifact manager."""
        return self.artifact_manager.save_iteration_summary_to_artifact(summary)
    
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
    
    def _admin_exit_detected(self) -> bool:
        """Check if Admin requested to exit the conversation."""
        # Check recent messages for admin exit commands
        for msg in reversed(self.groupchat.messages[-5:]):  # Check last 5 messages
            if isinstance(msg, dict) and msg.get("name") == "Admin":
                content = msg.get("content", "").lower()
                if any(term in content for term in self._exit_terms):
                    return True
        return False
    
    def compile_final_report(self, additional_info: Dict[str, Any] = None) -> str:
        """Compile the final report from all draft iterations."""
        if not self.draft_lineage:
            return "No report drafts were generated."
        
        # Get the latest draft
        latest_draft = self.draft_lineage[-1]
        
        # Extract the draft content
        if "content" in latest_draft:
            report = latest_draft["content"]
        else:
            report = "Report content not found."
        
        # Add metadata if available
        if additional_info:
            report += f"\n\n---\nAdditional Information:\n"
            for key, value in additional_info.items():
                report += f"- {key}: {value}\n"
        
        # Add summary of iterations if multiple
        if self.current_outer_turn > 1:
            report += f"\n\n---\nReport generated after {self.current_outer_turn} iterations.\n"
        
        return report
    
    def _process_messages_and_save_artifacts(self) -> Dict[str, Any]:
        """Process all messages and extract artifacts for saving."""
        # Scan and save any remaining code blocks
        self._scan_and_save_new_code()
        
        # Scan and save any remaining draft blocks
        self._scan_and_save_new_drafts()
        
        # Attach any pending planner feedback
        self._attach_planner_feedback()
        
        # Create results summary
        inner_results = {
            "completed": True,
            "code_created": len(self.code_lineage) > 0,
            "report_created": len(self.draft_lineage) > 0,
            "execution_success": self.check_execution_success(),
            "artifacts": {
                "codes": len(self.code_lineage),
                "drafts": len(self.draft_lineage),
            }
        }
        
        return inner_results
    
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
        
        # Get last planner message
        last_planner_msg = None
        for msg in reversed(self.groupchat.messages):
            if msg.get("name") == "Planner":
                last_planner_msg = msg.get("content", "").strip()
                break
        
        if not last_planner_msg:
            return
        
        # Add feedback to latest entries if they don't have it
        latest_code = self.artifact_manager.get_latest_code_entry()
        if latest_code and not latest_code.get("feedback"):
            self.artifact_manager.add_feedback_to_code(last_planner_msg)
        
        latest_draft = self.artifact_manager.get_latest_draft_entry()
        if latest_draft and not latest_draft.get("feedback"):
            self.artifact_manager.add_feedback_to_draft(last_planner_msg)
    
    def _count_existing_outer_iterations(self) -> int:
        """Count how many outer iterations have been completed previously."""
        max_iteration = 0
        
        print("\n🔍 DEBUG: Counting existing outer iterations...")
        
        if os.path.exists(self.config.artifact_dir):
            print(f"  📁 Artifact directory exists: {self.config.artifact_dir}")
            all_files = os.listdir(self.config.artifact_dir)
            print(f"  📄 Total files in artifact directory: {len(all_files)}")
            
            # Debug: print all files
            print(f"  📄 Files found:")
            for f in sorted(all_files):
                print(f"      - {f}")
            
            # Look for ANY artifact files that indicate an iteration was completed
            # This includes summarizer_output files, summary_outer files, etc.
            
            # Check summarizer output files FIRST (most reliable indicator)
            for filename in all_files:
                if filename.startswith("summarizer_output_") and filename.endswith(".txt"):
                    # Format: summarizer_output_{N}_{timestamp}.txt
                    try:
                        # Split by underscore and get the iteration number
                        parts = filename.replace("summarizer_output_", "").replace(".txt", "").split("_")
                        if parts and parts[0].isdigit():
                            iteration_num = int(parts[0])
                            print(f"    ✓ Found summarizer file: {filename} → Iteration {iteration_num}")
                            max_iteration = max(max_iteration, iteration_num)
                    except Exception as e:
                        print(f"    ❌ Error parsing {filename}: {e}")
                        continue
            
            # Also check for summary files (in case summarizer files are missing)
            for filename in all_files:
                if filename.startswith("summary_outer") and filename.endswith(".json"):
                    # Format: summary_outer{N}_{timestamp}.json
                    try:
                        parts = filename.replace("summary_outer", "").replace(".json", "").split("_")
                        if parts and parts[0].isdigit():
                            iteration_num = int(parts[0])
                            print(f"    ✓ Found summary file: {filename} → Iteration {iteration_num}")
                            max_iteration = max(max_iteration, iteration_num)
                    except Exception as e:
                        print(f"    ❌ Error parsing {filename}: {e}")
                        continue
            
            print(f"  📊 Maximum iteration found in artifacts: {max_iteration}")
        else:
            print(f"  ⚠️ Artifact directory does not exist: {self.config.artifact_dir}")
        
        print(f"  🎯 Highest completed iteration: {max_iteration}")
        print(f"  ➡️ Next iteration should be: {max_iteration + 1}")
        print(f"  📌 Returning: {max_iteration}")
        
        return max_iteration
    
    def run(self, initial_task: str) -> Dict[str, Any]:
        """Run the complete group chat with iteration control."""
        task = initial_task
        final_result = {}
        self.conversation_terminated = False
        
        print("\n" + "="*80)
        print("🚀 STARTING GROUP CHAT MANAGER RUN")
        print("="*80)
        print(f"  Initial outer_turn: {self.current_outer_turn}")
        print(f"  Max outer turns: {self.max_outer_turn}")
        print(f"  Max inner turns: {self.max_inner_turn}")
        
        # Check for existing artifacts
        print("\n📦 Checking for existing artifacts...")
        
        # Always count existing iterations to determine where to start
        existing_iterations = self._count_existing_outer_iterations()
        
        # Set the starting iteration correctly
        if existing_iterations > 0:
            self.current_outer_turn = existing_iterations
            print(f"\n🔄 Found {existing_iterations} completed iteration(s)")
            print(f"📊 Starting at iteration: {self.current_outer_turn + 1}")
        else:
            self.current_outer_turn = 0
            print(f"\n🆕 Starting fresh from iteration 1")
        
        # MULTIPLE OUTER ITERATIONS LOOP
        while self.current_outer_turn < self.max_outer_turn and not self.conversation_terminated:
            # Increment for the current iteration
            self.current_outer_turn += 1
            self.current_inner_turn = 1
            
            print(f"\n" + "="*80)
            print(f"🚀 STARTING OUTER ITERATION {self.current_outer_turn}/{self.max_outer_turn}")
            print("="*80)
            
            # Update artifact manager
            self.artifact_manager.set_iteration(self.current_outer_turn, self.current_inner_turn)
            
            # Reset message tracking for new iteration
            self.last_processed_message_index = 0
            
            # CRITICAL: Load context from ALL previous iterations
            context_message = ""
            
            # For ANY iteration, check if there are previous artifacts to load
            if self.current_outer_turn == 1 and existing_iterations > 0:
                # First iteration but there are existing artifacts from previous runs
                print("\n📚 Loading artifacts from previous session...")
                artifact_summary = self.iteration_manager.load_and_summarize_all_artifacts()
                if artifact_summary:
                    print("\n" + "=" * 80)
                    print("EXISTING ARTIFACTS FROM PREVIOUS SESSION")
                    print("=" * 80)
                    print(artifact_summary)
                    print("=" * 80)
                    context_message = f"{artifact_summary}\n\nBased on the existing artifacts found above, please analyze what has been done and determine the best approach for the current task."
            
            elif self.current_outer_turn > 1:
                # For iterations > 1, load context from ALL previous iterations in this session
                print("\n📚 Loading context from previous iterations...")
                
                # First, get the immediate previous iteration context
                prev_context = self.load_previous_iteration_context()
                if prev_context and "summary_text" in prev_context:
                    print("  ✅ Loaded previous iteration summary")
                    context_message = f"**Previous Iteration {self.current_outer_turn - 1} Summary:**\n{prev_context['summary_text']}\n\n"
                    
                    # Add metrics from previous iteration
                    if "metrics" in prev_context:
                        metrics = prev_context["metrics"]
                        context_message += f"**Previous Iteration Metrics:**\n"
                        context_message += f"- Codes created: {metrics.get('codes_created', 0)}\n"
                        context_message += f"- Drafts created: {metrics.get('drafts_created', 0)}\n"
                        context_message += f"- Last code success: {metrics.get('last_code_success', 'N/A')}\n\n"
                
                # Also check if we need to load context from even earlier iterations
                if self.current_outer_turn > 2:
                    # Load summary of ALL previous iterations
                    all_prev_summary = self.iteration_manager.get_all_previous_artifacts()
                    if all_prev_summary:
                        context_message = f"**Overview of All Previous Iterations:**\n{all_prev_summary}\n\n{context_message}"
                
                # Check for any artifacts from previous sessions if this is still early in the current session
                if self.current_outer_turn <= 2 and existing_iterations > 0:
                    # Also include context from previous sessions
                    print("  📂 Including context from previous session artifacts...")
                    prev_session_summary = self.iteration_manager.load_and_summarize_all_artifacts()
                    if prev_session_summary:
                        context_message = f"**Previous Session Work:**\n{prev_session_summary}\n\n{context_message}"
            
            # Build the task message for this iteration
            if context_message:
                if self.current_outer_turn == 1:
                    # First iteration - use the original task with context
                    iteration_task = f"""Context from previous work:
{context_message}

Current task:
{initial_task}

Please analyze what has been done and plan the next steps accordingly. If previous work exists and is satisfactory, you may focus on refinements or declare the task complete."""
                else:
                    # CRITICAL: For iterations > 1, PROMPT THE USER for specific instructions
                    print("\n" + "="*80)
                    print(f"📝 ITERATION {self.current_outer_turn} USER INPUT REQUIRED")
                    print("="*80)
                    
                    # Show the user the context
                    print("\n📊 Context from previous iterations:")
                    print("-" * 60)
                    print(context_message)
                    print("-" * 60)
                    
                    print(f"\n🎯 Original task: {initial_task}")
                    print(f"\n📌 This is iteration {self.current_outer_turn} of {self.max_outer_turn}")
                    
                    # Prompt user for specific instructions
                    print("\n" + "="*60)
                    print("Please provide your feedback and instructions for this iteration.")
                    print("You can:")
                    print("  1. Specify what to improve or fix")
                    print("  2. Request new features or analysis")
                    print("  3. Ask to focus on specific aspects")
                    print("  4. Type 'continue' to let agents decide improvements")
                    print("  5. Type 'approve' to accept current results and exit")
                    print("="*60)
                    
                    user_input = input("\n👤 Your instructions for iteration " + str(self.current_outer_turn) + ": ").strip()
                    
                    # Check if user wants to exit
                    if user_input.lower() in ['approve', 'approved', 'exit', 'quit', 'stop']:
                        print("\n✅ User approved current results. Ending iterations.")
                        self.conversation_terminated = True
                        break
                    
                    # Build the iteration task with user input
                    if user_input.lower() == 'continue' or not user_input:
                        # User wants agents to decide
                        iteration_task = f"""{context_message}Based on the previous iteration(s), please continue improving the solution.

Original task: {initial_task}

The user has asked you to continue and decide what improvements to make.
Focus on:
1. Addressing any issues or errors from the previous iteration
2. Improving code quality and error handling
3. Enhancing the report with more insights
4. Optimizing performance or adding new features

This is iteration {self.current_outer_turn} of {self.max_outer_turn}. Build upon the previous work."""
                    else:
                        # Use the specific user instructions
                        iteration_task = f"""{context_message}Based on the previous iteration(s), please improve the solution according to user feedback.

Original task: {initial_task}

**USER INSTRUCTIONS FOR THIS ITERATION:**
{user_input}

This is iteration {self.current_outer_turn} of {self.max_outer_turn}. Focus on addressing the user's specific requests above."""
                    
                    print(f"\n✅ User input received. Starting iteration {self.current_outer_turn}...")
            else:
                # No previous context, start fresh (shouldn't happen often)
                iteration_task = f"""{initial_task}

This is iteration {self.current_outer_turn} of {self.max_outer_turn}."""
            
            # Clear the last summarizer output for this new iteration
            self.last_summarizer_output = None
            
            # Initialize conversation for this iteration
            try:
                print(f"\n--- Starting iteration {self.current_outer_turn} conversation (max {self.max_inner_turn} rounds) ---")
                
                # Create a fresh group chat for this iteration (with message history cleared)
                self.groupchat.messages.clear()
                
                # Initiate chat for this iteration
                result = self.agents["user_proxy"].initiate_chat(
                    self.manager,
                    message=iteration_task,
                    clear_history=True,  # Clear agent history for each iteration
                    silent=False
                )
                
                print(f"\n✅ Iteration {self.current_outer_turn} completed successfully")
                
                # Check if user requested termination
                if self._admin_exit_detected():
                    print("\n🛑 User requested termination")
                    self.conversation_terminated = True
                    break
                
            except Exception as e:
                # Check if the exception indicates termination
                error_msg = str(e).lower()
                termination_keywords = ["terminating", "no reply generated", "user requested to end", 
                                       "exit", "stop", "terminate", "quit", "quit_debug"]
                
                if "quit_debug" in error_msg:
                    print("\n" + "="*80)
                    print("🛑 DEBUG QUIT REQUESTED")
                    print("="*80)
                    self._print_debug_state()
                    return {"report": "Debug quit requested.", "debug": True}
                    
                if any(keyword in error_msg for keyword in termination_keywords):
                    # This is expected when Summarizer terminates the inner loop
                    print(f"\n✅ Iteration {self.current_outer_turn} completed (terminated by Summarizer)")
                else:
                    print(f"\n⚠️ Iteration {self.current_outer_turn} ended with: {e}")
                
                # Check if user explicitly requested to stop
                if self._admin_exit_detected():
                    print("\n🛑 User requested termination")
                    self.conversation_terminated = True
                    break
            
            # After each iteration, save any remaining artifacts if needed
            if not self.last_summarizer_output:
                print(f"\n📝 Saving iteration {self.current_outer_turn} artifacts...")
                inner_results = self._process_messages_and_save_artifacts()
                if inner_results["code_created"] or inner_results["report_created"]:
                    summary = self.summarize_iteration(inner_results)
            
            # Check if we should continue to next iteration
            print(f"\n📊 Iteration {self.current_outer_turn} of {self.max_outer_turn} complete")
            if self.current_outer_turn >= self.max_outer_turn:
                print(f"🏁 Completed all {self.max_outer_turn} iterations")
                break
            else:
                # Ask user if they want to continue to next iteration
                print(f"\n" + "="*60)
                print(f"Iteration {self.current_outer_turn} completed.")
                print(f"Would you like to continue to iteration {self.current_outer_turn + 1}?")
                print("  - Type 'yes' or press Enter to continue")
                print("  - Type 'no' or 'exit' to stop")
                print("="*60)
                
                continue_choice = input("\n👤 Continue to next iteration? [yes]/no: ").strip().lower()
                
                if continue_choice in ['no', 'n', 'exit', 'quit', 'stop']:
                    print("\n✅ User chose to stop iterations.")
                    self.conversation_terminated = True
                    break
                else:
                    print(f"\n➡️ Continuing to iteration {self.current_outer_turn + 1}...")
                    # Small delay between iterations
                    import time
                    time.sleep(2)
        
        # Generate final report
        print("\n" + "="*80)
        print("📝 GENERATING FINAL REPORT")
        print("="*80)
        
        if self.draft_lineage:
            final_result = {
                "iteration": self.current_outer_turn,
                "report": self.compile_final_report({}),
                "completed": True,
                "total_iterations": self.current_outer_turn
            }
            print("✅ Final report generated successfully")
        else:
            final_result = {"report": "Conversation terminated without generating results.", "terminated": True}
            print("⚠️ No drafts were generated")
        
        return final_result

    def _print_debug_state(self):
        """Print comprehensive debug information about current state."""
        print("\n" + "="*60)
        print("DEBUG STATE INFORMATION")
        print("="*60)
        print(f"Current outer turn: {self.current_outer_turn}")
        print(f"Current inner turn: {self.current_inner_turn}")
        print(f"Artifact manager outer: {self.artifact_manager.current_outer_turn}")
        print(f"Artifact manager inner: {self.artifact_manager.current_inner_turn}")
        print(f"Code lineage count: {len(self.code_lineage)}")
        print(f"Draft lineage count: {len(self.draft_lineage)}")
        
        print(f"\nArtifact directory: {self.config.artifact_dir}")
        if os.path.exists(self.config.artifact_dir):
            files = os.listdir(self.config.artifact_dir)
            print(f"Files in artifact dir ({len(files)} total):")
            
            # Count specific artifact types
            json_summaries = [f for f in files if f.startswith("summary_outer") and f.endswith(".json")]
            summarizer_outputs = [f for f in files if f.startswith("summarizer_output") and f.endswith(".txt")]
            
            print(f"\n  📊 Artifact breakdown:")
            print(f"    - JSON summaries: {len(json_summaries)}")
            print(f"    - Summarizer outputs: {len(summarizer_outputs)}")
            
            # Extract iteration numbers from artifacts
            iterations_found = set()
            for f in files:
                if f.startswith("summary_outer"):
                    try:
                        parts = f.replace("summary_outer", "").replace(".json", "").split("_")
                        if parts and parts[0].isdigit():
                            iterations_found.add(int(parts[0]))
                    except:
                        pass
                elif f.startswith("summarizer_output_"):
                    try:
                        parts = f.replace("summarizer_output_", "").replace(".txt", "").split("_")
                        if parts and parts[0].isdigit():
                            iterations_found.add(int(parts[0]))
                    except:
                        pass
            
            if iterations_found:
                print(f"    - Iterations with artifacts: {sorted(iterations_found)}")
                print(f"    - Highest iteration completed: {max(iterations_found)}")
            
            print(f"\n  📄 All files:")
            for f in sorted(files)[:10]:  # Show first 10
                print(f"    - {f}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more files")
        else:
            print("  (directory does not exist)")
        
        print("="*60)
    
    def summarize_iteration(self, inner_results: Dict[str, Any]) -> Dict[str, Any]:
        """Use summarizer agent to create iteration summary and save to artifact."""
        print(f"\n📝 Creating iteration summary...")
        print(f"  🔍 Current outer_turn in summarize_iteration: {self.current_outer_turn}")
        print(f"  🔍 Artifact manager outer_turn: {self.artifact_manager.current_outer_turn}")
        
        # Ensure artifact manager has correct iteration
        self.artifact_manager.set_iteration(self.current_outer_turn, self.current_inner_turn)
        print(f"  ✅ Ensured artifact manager is at outer={self.current_outer_turn}, inner={self.current_inner_turn}")
        
        # Get stats from artifact manager
        stats = self.artifact_manager.get_iteration_stats()
        
        # Prepare summary data
        summary_data = {
            "outer_iteration": self.current_outer_turn,
            "total_inner_iterations": stats["inner_iteration"],
            "codes_created": stats["codes_created"],
            "drafts_created": stats["drafts_created"],
            "last_code_success": stats["last_code_success"],
            "completion_status": inner_results.get("completed", False)
        }
        
        print(f"  - Codes created: {summary_data['codes_created']}")
        print(f"  - Drafts created: {summary_data['drafts_created']}")
        
        # Use the captured summarizer output
        summary_response = ""
        if hasattr(self, 'last_summarizer_output') and self.last_summarizer_output:
            print(f"  ✅ Using captured Summarizer output")
            summary_response = self.last_summarizer_output
        else:
            print(f"  ⚠️ No captured summarizer output, checking artifact files...")
            
            # Try to load from the saved summarizer_output file for this iteration
            summarizer_file = None
            if os.path.exists(self.config.artifact_dir):
                for filename in sorted(os.listdir(self.config.artifact_dir), reverse=True):
                    if filename.startswith(f"summarizer_output_{self.current_outer_turn}_") and filename.endswith(".txt"):
                        summarizer_file = os.path.join(self.config.artifact_dir, filename)
                        break
            
            if summarizer_file and os.path.exists(summarizer_file):
                print(f"  ✅ Found summarizer file: {os.path.basename(summarizer_file)}")
                try:
                    with open(summarizer_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Extract the content between the markers
                        start_marker = "=" * 50 + "\n\n"
                        end_marker = "\n\n" + "=" * 50
                        start_idx = content.find(start_marker)
                        end_idx = content.rfind(end_marker)
                        if start_idx != -1 and end_idx != -1:
                            summary_response = content[start_idx + len(start_marker):end_idx].strip()
                            print(f"  ✅ Extracted summary from file")
                        else:
                            summary_response = content.strip()
                except Exception as e:
                    print(f"  ❌ Error reading summarizer file: {e}")
        
        # Build complete summary object with all data
        summary = {
            "iteration": self.current_outer_turn,
            "outer_iteration": self.current_outer_turn,
            "inner_turns": self.current_inner_turn,
            "artifacts": inner_results,
            "summary_text": summary_response,
            "metrics": summary_data,
            "code_lineage": self.artifact_manager.get_code_lineage(),
            "draft_lineage": self.artifact_manager.get_draft_lineage(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save JSON artifact
        print(f"  💾 Saving JSON artifact for iteration {self.current_outer_turn}")
        artifact_file = self.artifact_manager.save_iteration_summary_to_artifact(summary)
        print(f"  ✅ JSON artifact saved: {artifact_file}")
        
        # Verify files were created
        if os.path.exists(artifact_file):
            file_size = os.path.getsize(artifact_file)
            print(f"  ✅ JSON file verified ({file_size} bytes)")
        
        # Check for .txt file
        txt_file = None
        if os.path.exists(self.config.artifact_dir):
            for filename in os.listdir(self.config.artifact_dir):
                if filename.startswith(f"summarizer_output_{self.current_outer_turn}_") and filename.endswith(".txt"):
                    txt_file = os.path.join(self.config.artifact_dir, filename)
                    break
        
        if txt_file and os.path.exists(txt_file):
            txt_size = os.path.getsize(txt_file)
            print(f"  ✅ TXT file verified ({txt_size} bytes)")
        else:
            print(f"  ⚠️ TXT file not found for iteration {self.current_outer_turn}")
        
        # Clear the last summarizer output for next iteration
        self.last_summarizer_output = None
        
        return summary
    
    def _scan_and_save_new_code(self):
        """Scan for new code blocks in messages and save them."""
        msgs = self.groupchat.messages
        print(f"  🔍 Scanning {len(msgs)} messages for code...")
        print(f"  🔍 DEBUG: Current outer_turn={self.current_outer_turn}, inner_turn={self.current_inner_turn}")
        
        # Look for code blocks in messages
        for i in range(self.last_processed_message_index, len(msgs)):
            msg = msgs[i]
            if msg.get("name") == "Engineer":
                content = msg.get("content", "")
                # Extract Python code blocks
                code_blocks = self._extract_python_blocks(content)
                for code in code_blocks:
                    if code.strip():
                        # Check for duplicates
                        code_hash = self._hash(code)
                        if code_hash not in self._saved_code_hashes:
                            # Save new code iteration
                            filepath = self.artifact_manager.save_code_iteration(code, success=False)
                            print(f"    ✅ New code saved to {filepath}")
                            self._saved_code_hashes.add(code_hash)
                        else:
                            print(f"    ⏭️ Duplicate code block, not saving")
            
            # Check for termination command in any message
            if isinstance(msg, dict) and msg.get("name") == "Admin":
                content = msg.get("content", "").lower()
                if "terminate" in content or "exit" in content or "stop" in content:
                    print("  🛑 Termination command received")
                    self.conversation_terminated = True
                    break
        
        # Update the last processed index
        self.last_processed_message_index = len(msgs)

    def _scan_and_save_new_drafts(self):
        """Scan for new draft blocks in messages and save them."""
        msgs = self.groupchat.messages
        print(f"  🔍 Scanning {len(msgs)} messages for drafts...")
        print(f"  🔍 DEBUG: Current outer_turn={self.current_outer_turn}, inner_turn={self.current_inner_turn}")
        
        # Look for draft blocks in messages
        for i in range(self.last_processed_message_index, len(msgs)):
            msg = msgs[i]
            if msg.get("name") == "Writer":
                content = msg.get("content", "")
                # Extract Markdown draft blocks
                draft_blocks = self._extract_md_blocks(content)
                for draft in draft_blocks:
                    if draft.strip():
                        # Check for duplicates
                        draft_hash = self._hash(draft)
                        if draft_hash not in self._saved_draft_hashes:
                            # Save new draft iteration
                            filepath = self.artifact_manager.save_draft_iteration(draft)
                            print(f"    ✅ New draft saved to {filepath}")
                            self._saved_draft_hashes.add(draft_hash)
                        else:
                            print(f"    ⏭️ Duplicate draft block, not saving")
            
            # Check for termination command in any message
            if isinstance(msg, dict) and msg.get("name") == "Admin":
                content = msg.get("content", "").lower()
                if "terminate" in content or "exit" in content or "stop" in content:
                    print("  🛑 Termination command received")
                    self.conversation_terminated = True
                    break
        
        # Update the last processed index
        self.last_processed_message_index = len(msgs)