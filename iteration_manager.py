import os
import json
from typing import Dict, Any

class IterationManager:
    """Manages iteration context and history."""
    
    def __init__(self, config, memory_manager):
        self.config = config
        self.memory_manager = memory_manager
    
    def load_previous_iteration_context(self, current_outer_turn: int) -> Dict[str, Any]:
        """Load context from previous iteration's artifact."""
        target_iteration = current_outer_turn - 1
        
        print(f"  🔍 Looking for artifact from iteration {target_iteration}")
        
        # First try to get from memory manager
        history = self.memory_manager.get_iteration_history()
        
        if history:
            print(f"  📚 Found {len(history)} items in history")
            for item in history:
                if item.get("iteration") == target_iteration and "artifact_file" in item:
                    artifact_path = item["artifact_file"]
                    print(f"  📂 Found artifact reference: {artifact_path}")
                    if os.path.exists(artifact_path):
                        with open(artifact_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            print(f"  ✅ Loaded artifact from memory manager reference")
                            return data
                    else:
                        print(f"  ⚠️ Artifact file not found at: {artifact_path}")
        
        # Fallback: look for the most recent artifact file directly
        if os.path.exists(self.config.artifact_dir):
            print(f"  🔍 Fallback: searching in {self.config.artifact_dir}")
            
            artifact_files = []
            for filename in os.listdir(self.config.artifact_dir):
                if filename.startswith(f"summary_outer{target_iteration}_") and filename.endswith(".json"):
                    artifact_files.append(os.path.join(self.config.artifact_dir, filename))
                    print(f"    - Found matching file: {filename}")
            
            if artifact_files:
                artifact_files.sort()
                most_recent = artifact_files[-1]
                print(f"  📂 Using most recent artifact: {os.path.basename(most_recent)}")
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
    
    def build_context_message(self, prev_context: Dict[str, Any]) -> str:
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
    
    def get_all_previous_artifacts(self) -> str:
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
    
    def load_and_summarize_all_artifacts(self) -> str:
        """Load and create a comprehensive summary of all existing artifacts."""
        summary_parts = []
        
        # Check if artifact directory exists and has files
        if not os.path.exists(self.config.artifact_dir):
            return ""
        
        # Get all summary files
        summary_files = []
        for filename in os.listdir(self.config.artifact_dir):
            if filename.startswith("summary_outer") and filename.endswith(".json"):
                summary_files.append(os.path.join(self.config.artifact_dir, filename))
        
        if not summary_files:
            print("  ℹ️ No previous artifacts found in artifact directory")
            return ""
        
        # Sort files to process in order
        summary_files.sort()
        
        print(f"\n📚 Found {len(summary_files)} artifact(s) from previous sessions")
        summary_parts.append("=" * 60)
        summary_parts.append("SUMMARY OF PREVIOUS WORK")
        summary_parts.append("=" * 60)
        
        # Process each artifact
        for filepath in summary_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                outer_iter = data.get("outer_iteration", "?")
                timestamp = data.get("timestamp", "unknown")[:19]  # First 19 chars of timestamp
                summary_text = data.get("summary_text", "")
                
                # Extract key metrics
                metrics = data.get("metrics", {})
                codes_created = metrics.get("codes_created", 0)
                drafts_created = metrics.get("drafts_created", 0)
                completion_status = metrics.get("completion_status", False)
                
                # Extract last successful code and draft info
                code_info = ""
                if data.get("code_lineage"):
                    last_code = data["code_lineage"][-1]
                    if last_code.get("success"):
                        code_file = os.path.basename(last_code.get("file", ""))
                        code_info = f"\n    - Last successful code: {code_file}"
                
                draft_info = ""
                if data.get("draft_lineage"):
                    last_draft = data["draft_lineage"][-1]
                    draft_file = os.path.basename(last_draft.get("file", ""))
                    draft_info = f"\n    - Last draft: {draft_file}"
                
                # Build summary for this iteration
                summary_parts.append(f"\n📁 Outer Iteration {outer_iter} (Completed: {timestamp})")
                summary_parts.append(f"    - Created {codes_created} code file(s) and {drafts_created} draft(s)")
                summary_parts.append(f"    - Status: {'✅ Complete' if completion_status else '⚠️ Incomplete'}")
                
                if code_info:
                    summary_parts.append(code_info)
                if draft_info:
                    summary_parts.append(draft_info)
                
                # Add brief summary text if available
                if summary_text:
                    # Take first 200 chars of summary
                    brief_summary = summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
                    summary_parts.append(f"    - Summary: {brief_summary}")
                    
                print(f"  ✓ Loaded artifact from iteration {outer_iter}")
                
            except Exception as e:
                print(f"  ⚠️ Could not load {os.path.basename(filepath)}: {e}")
                continue
        
        # Look for special output files (like summarizer outputs)
        summarizer_files = []
        for filename in os.listdir(self.config.artifact_dir):
            if filename.startswith("summarizer_output") and filename.endswith(".txt"):
                summarizer_files.append(os.path.join(self.config.artifact_dir, filename))
        
        if summarizer_files:
            summary_parts.append("\n" + "=" * 60)
            summary_parts.append("ADDITIONAL SUMMARIES")
            summary_parts.append("=" * 60)
            
            # Get the most recent summarizer output
            summarizer_files.sort()
            latest_summarizer = summarizer_files[-1]
            
            try:
                with open(latest_summarizer, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Extract the main content between the markers
                    start_marker = "=" * 50 + "\n\n"
                    end_marker = "\n\n" + "=" * 50
                    
                    start_idx = content.find(start_marker)
                    end_idx = content.rfind(end_marker)
                    
                    if start_idx != -1 and end_idx != -1:
                        main_content = content[start_idx + len(start_marker):end_idx]
                        # Try to parse as JSON if possible
                        try:
                            json_data = json.loads(main_content)
                            if isinstance(json_data, dict) and "summary" in json_data:
                                summary_obj = json_data["summary"]
                                summary_parts.append("\n📊 Latest Detailed Summary:")
                                summary_parts.append(f"    Objective: {summary_obj.get('objective', 'N/A')}")
                                
                                if "data_tasks" in summary_obj:
                                    summary_parts.append("    Data Tasks Completed:")
                                    for task in summary_obj["data_tasks"][:3]:  # First 3 tasks
                                        summary_parts.append(f"      • {task[:100]}...")
                                
                                if "code_progress" in summary_obj:
                                    summary_parts.append(f"    Code Progress: {summary_obj['code_progress'][:150]}...")
                                
                                if "content_progress" in summary_obj:
                                    summary_parts.append(f"    Content Progress: {summary_obj['content_progress'][:150]}...")
                        except json.JSONDecodeError:
                            # Not JSON, just include first few lines
                            lines = main_content.split('\n')[:5]
                            summary_parts.append("\n📊 Latest Summary Notes:")
                            for line in lines:
                                if line.strip():
                                    summary_parts.append(f"    {line[:100]}...")
                
                print(f"  ✓ Included latest summarizer output")
                
            except Exception as e:
                print(f"  ⚠️ Could not read summarizer output: {e}")
        
        # Add recommendation for next steps
        summary_parts.append("\n" + "=" * 60)
        summary_parts.append("RECOMMENDATION")
        summary_parts.append("=" * 60)
        summary_parts.append("Based on the artifacts found, previous work exists on this task.")
        summary_parts.append("Please review the above summary and determine if you should:")
        summary_parts.append("  1. Continue from where the previous session left off")
        summary_parts.append("  2. Refine or improve the existing work")
        summary_parts.append("  3. Start fresh with a different approach")
        summary_parts.append("=" * 60)
        
        return "\n".join(summary_parts)
