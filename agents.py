import autogen
from typing import Dict, Any

class CustomAdminAgent(autogen.ConversableAgent):
    """Custom Admin agent with improved user prompts."""
    
    def get_human_input(self, prompt: str = "") -> str:
        """Override to provide better prompt message."""
        # Custom prompt that's clearer for the user
        custom_prompt = (
            "\n[Admin Input Required]\n"
            "Options:\n"
            "  - Provide feedback or instructions for the team\n"
            "  - Type 'approved' when satisfied with the results\n"
            "  - Type 'terminate' to end the conversation\n"
            "  - Type 'quit_debug' to exit with debug information\n"
            "  - Press Enter to let agents continue\n"
            "\nYour input: "
        )
        user_input = input(custom_prompt)
        
        # Check for debug quit
        if user_input.lower() == 'quit_debug':
            # Raise an exception with the debug quit message
            raise Exception("quit_debug requested by user")
        
        return user_input

def create_agents(config, llm_config: Dict[str, Any], exit_terms: set = None) -> Dict[str, autogen.ConversableAgent]:
    """Create all agents for the group chat.
    
    Args:
        config: Configuration object
        llm_config: LLM configuration dictionary
        exit_terms: Set of termination terms (optional)
    """
    
    # Use default exit terms if none provided
    if exit_terms is None:
        exit_terms = {"exit", "terminate", "stop", "end conversation", "quit", "approved", "bye", "goodbye"}
    
    agents = {}
    
    # User proxy/Admin agent with termination handling - using custom class
    agents["user_proxy"] = CustomAdminAgent(
        name="Admin",
        system_message=(
            "You are the Admin, the bridge between the user and the technical team.\n"
            "Your key responsibilities:\n\n"
            "1. UNDERSTAND USER REQUIREMENTS:\n"
            "   - Carefully analyze what the user is asking for\n"
            "   - Identify key objectives, metrics, and deliverables\n"
            "   - Clarify any ambiguous requirements\n"
            "   - Consider both explicit and implicit needs\n\n"
            "2. ELABORATE AND COMMUNICATE TO PLANNER:\n"
            "   - Break down the user's request into specific, actionable tasks\n"
            "   - Specify what data needs to be collected (time periods, metrics, sources)\n"
            "   - Define what analysis should be performed (comparisons, trends, calculations)\n"
            "   - Describe desired visualizations (chart types, key information to highlight)\n"
            "   - Outline report structure and key points to cover\n"
            "   - Provide context about the target audience and purpose\n\n"
            "3. QUALITY CONTROL:\n"
            "   - Review outputs to ensure they meet user requirements\n"
            "   - Check if all requested elements are addressed\n"
            "   - Verify data accuracy and visualization clarity\n"
            "   - Ensure the report is suitable for the intended audience\n\n"
            "4. ITERATIVE REFINEMENT:\n"
            "   - Provide specific, actionable feedback to improve outputs\n"
            "   - Focus feedback on gaps between current output and user requirements\n"
            "   - Guide the team toward the user's vision\n"
            "   - Balance technical accuracy with user accessibility\n\n"
            "5. COMPLETION:\n"
            "   - When outputs fully satisfy user requirements, respond with 'APPROVED'\n"
            "   - You can type 'terminate' at any time to end the conversation\n\n"
            "Remember: You represent the user's interests. Ensure the team understands not just WHAT "
            "to do, but WHY it matters and HOW it serves the user's goals."
        ),
        code_execution_config=False,
        llm_config=llm_config,
        human_input_mode="ALWAYS",
        is_termination_msg=list(exit_terms),  # Convert set to list for autogen
        max_consecutive_auto_reply=0,  # Ensure human input is always required
        default_auto_reply="",  # No auto-reply
        description="Admin who understands user requirements, elaborates them for the team, reviews outputs, and ensures quality. Type 'exit' to end."
    )
    
    # Planner agent
    agents["planner"] = autogen.ConversableAgent(
        name="Planner",
        system_message=(
            "You are the Planner. Your responsibilities:\n"
            "1. Determine what information is needed to complete the task\n"
            "2. IMPORTANT: Always delegate to Engineer when you need:\n"
            "   - Stock data from Yahoo Finance or any other API\n"
            "   - Historical price data, volume, or market metrics\n"
            "   - Charts, figures, or visualizations\n"
            "   - Data analysis or calculations\n"
            "   - Any information that requires code execution\n"
            "3. Provide clear, specific requirements to Engineer including:\n"
            "   - What data to fetch (ticker symbols, date ranges, metrics)\n"
            "   - What visualizations to create (chart types, labels, colors)\n"
            "   - What analysis to perform (calculations, comparisons, trends)\n"
            "4. After Engineer completes coding and Executor runs it:\n"
            "   - Review the results and data quality\n"
            "   - Verify factual accuracy\n"
            "   - Request additional data or figures if needed\n"
            "   - IMPORTANT: Check for any interactive commands (plt.show(), input(), etc.) and request Engineer to remove them\n"
            "5. Once data is collected, instruct Writer to create the report\n"
            "6. Review Writer's content for:\n"
            "   - SEO optimization and web discoverability\n"
            "   - Factual accuracy against collected data\n"
            "   - Completeness and clarity\n"
            "7. If code fails, provide specific feedback to Engineer for fixes\n"
            "8. When satisfied with both code and report, respond with 'REVIEW_COMPLETE'\n\n"
            "WORKFLOW: Task → Call Engineer for data/figures → Review results → "
            "Call Writer for report → Review report → Iterate or complete"
        ),
        description=(
            "Planner. Determines information needed, MUST delegate to Engineer for all data fetching "
            "and visualization tasks, reviews progress, provides feedback on code and writing quality."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    # Engineer agent
    agents["engineer"] = autogen.AssistantAgent(
        name="Engineer",
        system_message=(
            "You are the Engineer. Your primary responsibilities:\n"
            "1. Write Python code to fetch data and create visualizations as requested by Planner\n"
            "2. IMPORTANT: When you receive previous code with feedback:\n"
            "   - Read the feedback carefully\n"
            "   - Address ALL issues mentioned\n"
            "   - Keep what works, fix what doesn't\n"
            "   - Add improvements suggested by Planner\n"
            "3. Common tasks you will handle:\n"
            "   - Fetch stock data using yfinance library (import yfinance as yf)\n"
            "   - Get historical prices: yf.Ticker(symbol).history(period='1mo')\n"
            "   - Calculate metrics: returns, volatility, moving averages\n"
            "   - Create visualizations using matplotlib or plotly\n"
            "   - Save data to CSV files for reference\n"
            "4. Code requirements:\n"
            "   - Follow SOLID principles\n"
            "   - Include proper error handling with try/except blocks\n"
            "   - Add logging statements to track progress\n"
            "   - Use descriptive variable names and comments\n"
            "5. IMPORTANT - File paths (you are already in the 'coding' directory):\n"
            "   - Save raw data to 'data/' subfolder (NOT 'coding/data/')\n"
            "   - Save visualizations to 'figures/' subfolder (NOT 'coding/figures/')\n"
            "   - Use RELATIVE paths only: 'data/file.csv', 'figures/chart.png'\n"
            "   - Create directories if they don't exist using os.makedirs(exist_ok=True)\n"
            "6. CRITICAL - NO INTERACTIVE COMMANDS:\n"
            "   - NEVER use plt.show() - it blocks execution\n"
            "   - NEVER use input() or any user input commands\n"
            "   - NEVER use interactive backends or GUI elements\n"
            "   - Always save figures with plt.savefig() instead of showing them\n"
            "   - Use matplotlib.use('Agg') at the start to set non-interactive backend\n"
            "7. Always include in your code:\n"
            "   - Import statements at the top\n"
            "   - Main execution in a function or script structure\n"
            "   - Print statements showing fetched data summary\n"
            "   - Save both data and visualizations\n"
            "8. Example code structure for stock data:\n"
            "   ```python\n"
            "   import matplotlib\n"
            "   matplotlib.use('Agg')  # Use non-interactive backend\n"
            "   import yfinance as yf\n"
            "   import pandas as pd\n"
            "   import matplotlib.pyplot as plt\n"
            "   from datetime import datetime, timedelta\n"
            "   import os\n"
            "   \n"
            "   # Create directories if needed\n"
            "   os.makedirs('data', exist_ok=True)\n"
            "   os.makedirs('figures', exist_ok=True)\n"
            "   \n"
            "   # Fetch data\n"
            "   ticker = yf.Ticker('NVDA')\n"
            "   data = ticker.history(period='1mo')\n"
            "   \n"
            "   # Save data (relative path - we're already in coding/)\n"
            "   data.to_csv('data/nvda_data.csv')\n"
            "   print(f'Data saved to data/nvda_data.csv')\n"
            "   \n"
            "   # Create visualization\n"
            "   plt.figure(figsize=(12, 6))\n"
            "   plt.plot(data.index, data['Close'])\n"
            "   plt.title('NVIDIA Stock Price - Last Month')\n"
            "   plt.savefig('figures/nvda_price.png', dpi=100, bbox_inches='tight')\n"
            "   plt.close()  # Clean up the figure\n"
            "   print(f'Figure saved to figures/nvda_price.png')\n"
            "   # NO plt.show() - we save instead!\n"
            "   ```"
        ),
        description=(
            "Engineer that writes code for data fetching (especially from Yahoo Finance), "
            "analysis, and visualization based on Planner's requirements. Iterates based on feedback."
        ),
        llm_config=llm_config,
    )
    
    # Executor agent
    agents["executor"] = autogen.ConversableAgent(
        name="Executor",
        system_message=(
            "You are the Executor. Execute the code written by the Engineer and report results clearly.\n"
            "CRITICAL: Before executing any code:\n"
            "1. Scan for interactive commands like plt.show(), input(), getpass(), etc.\n"
            "2. If found, automatically remove or comment them out\n"
            "3. Replace plt.show() with plt.savefig() if a filename isn't already specified\n"
            "4. Add matplotlib.use('Agg') at the start if matplotlib is imported\n"
            "\n"
            "When executing code:\n"
            "1. Report what data was successfully fetched\n"
            "2. List all files created (data files, figures)\n"
            "3. Show key statistics or findings from the data\n"
            "4. Provide full error traceback if code fails\n"
            "5. If you had to modify the code to remove interactive elements, mention it\n"
            "\n"
            "Example of fixing interactive code:\n"
            "If you see: plt.show()\n"
            "Comment it out: # plt.show()  # Removed interactive display\n"
            "And ensure there's a savefig before it."
        ),
        human_input_mode="NEVER",
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": config.coding_dir,
            "use_docker": False,
        },
    )
    
    # Writer agent
    agents["writer"] = autogen.ConversableAgent(
        name="Writer",
        system_message=(
            "You are the Writer. Create engaging, well-structured blog posts in markdown format.\n"
            "1. IMPORTANT: When you receive a previous draft with feedback:\n"
            "   - Read the feedback carefully\n"
            "   - Keep the good parts of the previous draft\n"
            "   - Improve sections mentioned in feedback\n"
            "   - Add new content as requested\n"
            "2. Writing requirements:\n"
            "   - Include relevant titles, sections, and formatting\n"
            "   - ALWAYS wrap your ENTIRE markdown content in ```md``` code blocks\n"
            "   - The markdown block should contain the complete blog post\n"
            "   - Incorporate data, charts, and findings from the code execution\n"
            "   - Reference figures using relative paths: ![Alt text](figures/filename.png)\n"
            "3. Format example (FOLLOW THIS EXACTLY):\n"
            "   ```md\n"
            "   # Your Blog Title Here\n"
            "   \n"
            "   ## Introduction\n"
            "   Your introduction content here...\n"
            "   \n"
            "   ## Analysis\n"
            "   Your analysis content here...\n"
            "   \n"
            "   ![Stock Chart](figures/stock_chart.png)\n"
            "   \n"
            "   ## Conclusion\n"
            "   Your conclusion here...\n"
            "   ```\n"
            "4. SEO optimization:\n"
            "   - Use relevant keywords naturally\n"
            "   - Include meta descriptions\n"
            "   - Create engaging headlines\n"
            "   - Structure with proper H1, H2, H3 tags\n"
            "5. Take feedback from Admin and Planner to refine your blog\n"
            "6. Each iteration should be better than the last\n"
            "7. DO NOT include 'filename:' directives - just the markdown in code blocks"
        ),
        description=(
            "Writer. Creates blogs based on code execution results and "
            "iterates based on feedback to refine the content."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    # Summarizer agent
    agents["summarizer"] = autogen.ConversableAgent(
        name="Summarizer",
        system_message=(
            "You are the Summarizer. After each outer iteration:\n"
            "1. Summarize key findings and progress made\n"
            "2. List action items and feedback received\n"
            "3. Highlight improvements made to code and writing\n"
            "4. Create a structured summary for memory persistence\n"
            "Format your summary as JSON-compatible text for artifact storage."
        ),
        description="Summarizer that creates iteration summaries for memory persistence.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    return agents
