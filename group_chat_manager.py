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
                self.custom_manager.current_inner_turn = max(1, (self.message_count + 3) // 4)
                # CRITICAL FIX: Always update artifact manager's iteration with current values
                self.custom_manager.artifact_manager.set_iteration(
                    self.custom_manager.current_outer_turn,
                    self.custom_manager.current_inner_turn
                )
        
        # Extract message content and sender name
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        sender_name = sender.name if hasattr(sender, 'name') else str(sender)
        
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
                    self.custom_manager.artifact_manager.save_summarizer_output(content)
                    self.custom_manager.last_summarizer_output = content
            
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
            
            # Look for summary files to determine the highest iteration number
            summary_files = []
            for filename in all_files:
                if filename.startswith("summary_outer") and filename.endswith(".json"):
                    summary_files.append(filename)
                    # Extract iteration number from filename
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
            
            print(f"  📊 Summary files found: {len(summary_files)}")
            
            # Also check summarizer output files
            summarizer_files = []
            for filename in all_files:
                if filename.startswith("summarizer_output_") and filename.endswith(".txt"):
                    summarizer_files.append(filename)
                    # Format: summarizer_output_{N}_{timestamp}.txt
                    try:
                        parts = filename.replace("summarizer_output_", "").replace(".txt", "").split("_")
                        if parts and parts[0].isdigit():
                            iteration_num = int(parts[0])
                            print(f"    ✓ Found summarizer file: {filename} → Iteration {iteration_num}")
                            max_iteration = max(max_iteration, iteration_num)
                    except Exception as e:
                        print(f"    ❌ Error parsing {filename}: {e}")
                        continue
            
            print(f"  📊 Summarizer files found: {len(summarizer_files)}")
        else:
            print(f"  ⚠️ Artifact directory does not exist: {self.config.artifact_dir}")
        
        print(f"  🎯 Maximum iteration found: {max_iteration}")
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
        
        # Check for existing artifacts BEFORE starting any iterations
        artifact_summary = ""
        if self.current_outer_turn == 0:  # Only on first run
            print("\n📦 Checking for existing artifacts...")
            artifact_summary = self.iteration_manager.load_and_summarize_all_artifacts()
            if artifact_summary:
                print("\n" + "=" * 80)
                print("EXISTING ARTIFACTS DETECTED")
                print("=" * 80)
                print(artifact_summary)
                print("=" * 80)
                
                # IMPORTANT: Count existing outer iterations to continue from the right number
                existing_iterations = self._count_existing_outer_iterations()
                if existing_iterations > 0:
                    # Set to existing iterations so the NEXT increment goes to existing+1
                    self.current_outer_turn = existing_iterations
                    print(f"\n🔄 ADJUSTED current_outer_turn to: {self.current_outer_turn}")
                    print(f"📊 Will start next iteration at: {self.current_outer_turn + 1}")
                    
                    # DON'T pre-update the artifact manager here - let the loop handle it
                    print(f"🔧 Artifact manager will be updated when loop starts")
        else:
            print(f"  ⏭️ Skipping artifact check (current_outer_turn={self.current_outer_turn})")
        
        # Add quit check before main loop
        print("\n💡 TIP: Type 'quit_debug' when prompted to exit with debug info")
        
        while not self.conversation_terminated and self.current_outer_turn < self.max_outer_turn:
            # Store the previous value for debugging
            prev_outer = self.current_outer_turn
            
            self.current_outer_turn += 1
            self.current_inner_turn = 1
            
            print(f"\n{'='*80}")
            print(f"🔄 INCREMENTING OUTER ITERATION")
            print(f"  Before increment: outer_turn was {prev_outer}")
            print(f"  After increment: outer_turn is now {self.current_outer_turn}")
            print(f"  Artifact manager before update: outer={self.artifact_manager.current_outer_turn}, inner={self.artifact_manager.current_inner_turn}")
            print(f"{'='*80}")
            
            # CRITICAL: Update artifact manager's iteration immediately after incrementing
            self.artifact_manager.set_iteration(self.current_outer_turn, self.current_inner_turn)
            print(f"✅ After set_iteration call:")
            print(f"   - GroupChatManager outer_turn: {self.current_outer_turn}")
            print(f"   - ArtifactManager outer_turn: {self.artifact_manager.current_outer_turn}")
            
            # VERIFY the update worked
            if self.artifact_manager.current_outer_turn != self.current_outer_turn:
                print(f"❌ ERROR: Artifact manager not synced! Expected {self.current_outer_turn}, got {self.artifact_manager.current_outer_turn}")
            
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
                    clear_history=False,
                    silent=False
                )
            except Exception as e:
                # Check if the exception indicates termination
                error_msg = str(e).lower()
                # Check for various termination indicators in the error message
                termination_keywords = ["terminating", "no reply generated", "user requested to end", 
                                       "exit", "stop", "terminate", "quit", "quit_debug"]
                
                if "quit_debug" in error_msg or (hasattr(e, 'args') and any("quit_debug" in str(arg).lower() for arg in e.args)):
                    print("\n" + "="*80)
                    print("🛑 DEBUG QUIT REQUESTED")
                    print("="*80)
                    self._print_debug_state()
                    return {"report": "Debug quit requested.", "debug": True}
                    
                if any(keyword in error_msg for keyword in termination_keywords):
                    self.conversation_terminated = True
                    print("\n🛑 Conversation terminated by user.")
                else:
                    print(f"Chat completed or interrupted: {e}")
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
            print(f"\n📝 About to summarize iteration {self.current_outer_turn}")
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
            for f in sorted(files)[:10]:  # Show first 10
                print(f"  - {f}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
        else:
            print("  (directory does not exist)")
        
        print(f"\nMemory manager history: {len(self.memory_manager.get_iteration_history())} entries")
        for entry in self.memory_manager.get_iteration_history()[-3:]:  # Last 3
            print(f"  - Iteration {entry.get('iteration')}: {entry.get('timestamp', 'N/A')[:19]}")
        
        print("="*60)
    
    def summarize_iteration(self, inner_results: Dict[str, Any]) -> Dict[str, Any]:
        """Use summarizer agent to create iteration summary and save to artifact."""
        print(f"\n📝 Creating iteration summary...")
        print(f"  🔍 DEBUG: Current outer_turn in summarize_iteration: {self.current_outer_turn}")
        print(f"  🔍 DEBUG: Artifact manager outer_turn: {self.artifact_manager.current_outer_turn}")
        
        # Ensure artifact manager has correct iteration
        self.artifact_manager.set_iteration(self.current_outer_turn, self.current_inner_turn)
        print(f"  ✅ Ensured artifact manager is at outer={self.current_outer_turn}, inner={self.current_inner_turn}")
        
        # Get stats from artifact manager
        stats = self.artifact_manager.get_iteration_stats()
        
        # Prepare summary data
        summary_data = {
            "outer_iteration": stats["outer_iteration"],
            "total_inner_iterations": stats["inner_iteration"],
            "codes_created": stats["codes_created"],
            "drafts_created": stats["drafts_created"],
            "last_code_success": stats["last_code_success"],
            "completion_status": inner_results.get("completed", False)
        }
        
        print(f"  - Codes created: {summary_data['codes_created']}")
        print(f"  - Drafts created: {summary_data['drafts_created']}")
        
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
            "code_lineage": self.artifact_manager.get_code_lineage(),
            "draft_lineage": self.artifact_manager.get_draft_lineage(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save to artifact
        print(f"  💾 About to save artifact for iteration {self.current_outer_turn}")
        artifact_file = self.artifact_manager.save_iteration_summary_to_artifact(summary)
        print(f"  ✅ Artifact file path: {artifact_file}")
        
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