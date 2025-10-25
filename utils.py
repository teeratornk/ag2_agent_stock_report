import os
import shutil
from typing import List, Optional
from datetime import datetime

def clean_working_directory(path: str, extensions: Optional[List[str]] = None):
    """Clean working directory by removing files with specified extensions."""
    if not os.path.exists(path):
        return
    
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            if extensions is None or any(file.endswith(ext) for ext in extensions):
                os.remove(file_path)

def archive_outputs(config, timestamp: Optional[str] = None):
    """Archive current outputs to a timestamped folder."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    archive_dir = os.path.join(config.output_dir, f"archive_{timestamp}")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Archive coding files
    if os.path.exists(config.coding_dir):
        shutil.copytree(
            config.coding_dir, 
            os.path.join(archive_dir, "coding"),
            dirs_exist_ok=True
        )
    
    # Archive draft files
    if os.path.exists(config.draft_dir):
        shutil.copytree(
            config.draft_dir,
            os.path.join(archive_dir, "draft"),
            dirs_exist_ok=True
        )
    
    return archive_dir

def extract_code_blocks(text: str, language: str = "python") -> List[str]:
    """Extract code blocks from markdown text."""
    blocks = []
    lines = text.split("\n")
    in_block = False
    current_block = []
    
    for line in lines:
        if line.strip().startswith(f"```{language}"):
            in_block = True
            current_block = []
        elif line.strip() == "```" and in_block:
            in_block = False
            if current_block:
                blocks.append("\n".join(current_block))
        elif in_block:
            current_block.append(line)
    
    return blocks

def save_figure(figure, filename: str, config):
    """Save a matplotlib figure to the appropriate directory."""
    figures_dir = config.get_subdir_path("figures")
    if not figures_dir:
        figures_dir = os.path.join(config.coding_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
    
    filepath = os.path.join(figures_dir, filename)
    figure.savefig(filepath, dpi=300, bbox_inches='tight')
    return filepath

def format_markdown_report(title: str, sections: dict) -> str:
    """Format a report in markdown with proper structure."""
    md_content = f"# {title}\n\n"
    md_content += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    for section_title, section_content in sections.items():
        md_content += f"## {section_title}\n\n"
        md_content += f"{section_content}\n\n"
    
    return md_content
