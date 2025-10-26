import os
import sys
import argparse
from datetime import datetime
import shutil
from config import Config, llm_config
from agents import create_agents
from group_chat_manager import CustomGroupChatManager
from memory_manager import MemoryManager
import glob

def clear_directory(path: str, preserve_gitkeep: bool = True):
    """Clear all contents of a directory but keep the directory itself."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            if preserve_gitkeep and filename == '.gitkeep':
                continue
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def setup_directories(config: Config, clear_artifacts: bool = False):
    """Create necessary directories for the project and clear working directories."""
    # Create all necessary directories (removed memory_dir)
    dirs = [config.coding_dir, config.draft_dir, config.artifact_dir, config.output_dir]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ Ensured directory exists: {dir_path}")
    
    # Always clear coding and draft folders
    print("\nClearing working directories...")
    clear_directory(config.coding_dir)
    print(f"  ✓ Cleared {config.coding_dir}/")
    clear_directory(config.draft_dir)
    print(f"  ✓ Cleared {config.draft_dir}/")
    
    # Optionally clear artifact folder
    if clear_artifacts:
        clear_directory(config.artifact_dir)
        print(f"  ✓ Cleared {config.artifact_dir}/")
    else:
        print(f"  ✓ Preserved {config.artifact_dir}/")
    
    # Never clear output directory
    print(f"  ✓ Preserved {config.output_dir}/")
    
    # Ensure subdirectories exist in coding folder
    config.ensure_working_subdirs()
    print(f"  ✓ Created subdirectories in {config.coding_dir}/")

def get_clear_artifacts_preference() -> bool:
    """Ask user if they want to clear artifacts."""
    print("\nArtifact Management:")
    print("  Artifacts contain memory from previous sessions.")
    print("  1. Preserve artifacts (continue from previous sessions)")
    print("  2. Clear artifacts (start fresh)")
    
    while True:
        choice = input("\nSelect option (1-2) [default: 1]: ").strip() or "1"
        if choice == "1":
            return False
        elif choice == "2":
            confirm = input("Are you sure you want to clear all artifacts? (y/n): ").strip().lower()
            return confirm == 'y'
        else:
            print("Invalid option. Please select 1 or 2.")

def get_user_task(default_task: str) -> str:
    """Get task from user interactively or use default."""
    print("\n" + "=" * 80)
    print("TASK SELECTION")
    print("=" * 80)
    print("\nDefault task:")
    print(f"  {default_task}")
    print("\nOptions:")
    print("  1. Use default task")
    print("  2. Enter custom task")
    print("  3. Exit")
    
    while True:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            print("\nUsing default task.")
            return default_task
        elif choice == "2":
            print("\nEnter your custom task:")
            print("(For multi-line input, type 'MULTI' on a new line, then end with 'END')")
            
            first_line = input("> ").strip()
            
            # Check if user wants multi-line input
            if first_line.upper() == 'MULTI':
                print("Enter your task (type 'END' on a new line when done):")
                lines = []
                while True:
                    line = input("> ")
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                custom_task = "\n".join(lines).strip()
            else:
                # Single line input (most common case)
                custom_task = first_line
            
            if custom_task:
                print(f"\nCustom task accepted:")
                print(f"  {custom_task[:100]}..." if len(custom_task) > 100 else f"  {custom_task}")
                confirm = input("\nConfirm this task? (y/n): ").strip().lower()
                if confirm == 'y':
                    return custom_task
                else:
                    print("Let's try again.")
            else:
                print("No task entered. Please try again.")
        elif choice == "3":
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid option. Please select 1, 2, or 3.")

def get_latest_draft_file(draft_dir: str) -> str:
    """Find the most recently modified file in the draft directory."""
    # Look for all files in draft directory
    files = []
    for pattern in ['*.md', '*.txt', '*']:
        files.extend(glob.glob(os.path.join(draft_dir, pattern)))
    
    # Filter out directories and hidden files
    files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith('.')]
    
    if not files:
        return None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def main():
    parser = argparse.ArgumentParser(
        description="Stock Report Generator with Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode (default)
  python main.py --task "Your task"       # Run with custom task
  python main.py --clear-artifacts        # Clear artifacts and start fresh
  python main.py --preserve-all           # Debug mode - preserve all directories
  python main.py --max-inner-turn 5       # Limit inner iterations to 5
  python main.py --max-outer-turn 2       # Limit outer iterations to 2
        """
    )
    parser.add_argument("--task", type=str, help="Custom task to execute")
    parser.add_argument("--max-inner-turn", type=int, default=100, help="Maximum inner iterations (default: 100)")
    parser.add_argument("--max-outer-turn", type=int, default=5, help="Maximum outer iterations (default: 3)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for task selection")
    parser.add_argument("--clear-artifacts", action="store_true", help="Clear artifact folder at start (default: preserve)")
    parser.add_argument("--preserve-all", action="store_true", help="Preserve all folders including coding and draft (for debugging)")
    
    args = parser.parse_args()
    
    # Default task
    default_task = (
        "Write a blogpost about the stock price performance of "
        "Nvidia in the past month. Today's date is 2025-10-23."
    )
    
    # Determine if we should clear artifacts (interactive mode if not specified via CLI)
    clear_artifacts = args.clear_artifacts
    
    # Determine task based on arguments
    if args.task:
        # Task provided via command line
        task = args.task
        print(f"\nUsing task from command line argument")
    elif args.interactive or (not args.task and sys.stdin.isatty()):
        # Interactive mode: explicitly requested or no task provided and running in terminal
        # First, ask about artifacts if not specified via command line
        if not args.clear_artifacts and not args.preserve_all:
            clear_artifacts = get_clear_artifacts_preference()
        
        # Then get the task
        task = get_user_task(default_task)
    else:
        # Non-interactive mode with no task provided
        task = default_task
        print(f"\nUsing default task")
    
    # Setup configuration and directories
    config = Config()
    valid, message = config.validate()
    if not valid:
        print(f"Configuration error: {message}")
        sys.exit(1)
    
    # Setup directories and clear as needed
    if args.preserve_all:
        print("\nPreserving all existing directories (debug mode)")
        # Just ensure directories exist without clearing
        dirs = [config.coding_dir, config.draft_dir, config.artifact_dir, config.output_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        config.ensure_working_subdirs()
    else:
        setup_directories(config, clear_artifacts=clear_artifacts)
    
    # Initialize memory manager
    memory_manager = MemoryManager(config)
    
    # Show memory status if artifacts were preserved
    if not clear_artifacts and not args.preserve_all:
        history = memory_manager.get_iteration_history()
        if history:
            print(f"\n📚 Found {len(history)} previous iteration(s) in memory")
            last_iter = history[-1]
            print(f"  Last iteration: {last_iter.get('timestamp', 'Unknown')[:19]}")
        else:
            print("\n📚 No previous iterations found (starting fresh)")
        
        # Check for any artifact files
        artifact_count = 0
        if os.path.exists(config.artifact_dir):
            for filename in os.listdir(config.artifact_dir):
                if (filename.startswith("summary_outer") and filename.endswith(".json")) or \
                   (filename.startswith("summarizer_output") and filename.endswith(".txt")):
                    artifact_count += 1
        
        if artifact_count > 0:
            print(f"  📁 Found {artifact_count} artifact file(s) in artifact directory")
            print("  These will be analyzed and summarized for the Planner at the start of execution")
    
    # Define exit terms in one place
    exit_terms = {"exit", "terminate", "stop", "end conversation", "quit", "approved", "bye", "goodbye"}
    
    # Create agents with exit_terms
    agents = create_agents(config, llm_config, exit_terms)
    
    # Create custom group chat manager with exit_terms
    chat_manager = CustomGroupChatManager(
        agents=agents,
        config=config,
        memory_manager=memory_manager,
        max_inner_turn=args.max_inner_turn,
        max_outer_turn=args.max_outer_turn,
        exit_terms=exit_terms
    )
    
    # Start the conversation
    print("\n" + "=" * 80)
    print("STARTING TASK EXECUTION")
    print("=" * 80)
    print(f"\nTask: {task}")
    print(f"Max inner turns: {args.max_inner_turn}")
    print(f"Max outer turns: {args.max_outer_turn}")
    print("-" * 80)
    
    final_result = chat_manager.run(task)
    
    # Save final output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.output_dir, f"report_{timestamp}.md")
    
    # Check if we have a proper report or just a termination message
    if final_result and "report" in final_result:
        report_content = final_result["report"]
        
        # Check if this is just a termination message
        if "Conversation terminated" in report_content or len(report_content) < 100:
            print("\nConversation terminated. Looking for draft report...")
            
            # Try to find the latest file in draft folder
            latest_draft = get_latest_draft_file(config.draft_dir)
            
            if latest_draft:
                print(f"Found draft file: {os.path.basename(latest_draft)}")
                
                # Read the draft file
                with open(latest_draft, "r", encoding="utf-8") as f:
                    report_content = f.read()
                
                # Save to output
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report_content)
                print(f"✓ Draft report copied to: {output_file}")
            else:
                print("No draft files found in draft folder.")
                
                # As a fallback, check coding folder for any report files
                coding_files = glob.glob(os.path.join(config.coding_dir, "*.md"))
                if coding_files:
                    latest_coding = max(coding_files, key=os.path.getmtime)
                    print(f"Found file in coding folder: {os.path.basename(latest_coding)}")
                    
                    with open(latest_coding, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(report_content)
                    print(f"✓ Report from coding folder saved to: {output_file}")
                else:
                    # Save whatever we have
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(report_content)
                    print(f"Warning: Could not find draft report. Saved termination message to: {output_file}")
        else:
            # We have a proper report
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"\n✓ Final report saved to: {output_file}")
    else:
        # No report in final_result, try to find draft
        print("\nNo report in final result. Looking for draft report...")
        
        latest_draft = get_latest_draft_file(config.draft_dir)
        
        if latest_draft:
            print(f"Found draft file: {os.path.basename(latest_draft)}")
            
            # Copy the draft file to output
            shutil.copy2(latest_draft, output_file)
            print(f"✓ Draft report copied to: {output_file}")
        else:
            print("Warning: No draft files found. No report generated.")
    
    print("\nTask completed successfully!")

if __name__ == "__main__":
    main()
