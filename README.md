# 📊 Stock Report Generator - Multi-Agent System

A smart application that automatically generates professional blog posts about stock performance using AI agents that work together like a team.

## 🎯 What Does This Do?

This application creates detailed, well-researched blog posts about stock market performance. It's like having a team of experts working together:

- **🎩 Admin** - You! The supervisor who provides the task and reviews the work
- **📋 Planner** - Organizes the work and decides what needs to be done
- **💻 Engineer** - Writes code to fetch real stock data and create charts
- **⚙️ Executor** - Runs the code and reports the results
- **✍️ Writer** - Creates professional blog posts with the data
- **📝 Summarizer** - Keeps track of progress and creates summaries

## 🚀 Quick Start Guide

### Prerequisites (What You Need First)

1. **Python 3.8 or newer** - [Download Python](https://www.python.org/downloads/)
2. **Azure OpenAI API Access** - You need an Azure account with OpenAI service enabled
3. **Basic command line knowledge** - Know how to open Terminal/Command Prompt

### Step 1: Download the Project

```bash
# Clone this repository (or download as ZIP)
git clone [your-repository-url]
cd report_stock_rev2
```

### Step 2: Install Required Packages

```bash
# Install AG2 (AutoGen) framework
pip install ag2

# Install other required packages
pip install python-dotenv yfinance matplotlib pandas

# Or install all at once from requirements file
pip install -r requirements.txt
```

### Step 3: Set Up Your API Keys

1. Copy the example configuration file:
```bash
cp .env.example .env
```

2. Open `.env` file in a text editor and add your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_MODEL=gpt-4
```

> 📌 **Where to find these values:**
> - Log into [Azure Portal](https://portal.azure.com)
> - Go to your Azure OpenAI resource
> - Find "Keys and Endpoint" in the left menu
> - Copy Key 1 or Key 2 for `AZURE_OPENAI_API_KEY`
> - Copy the Endpoint for `AZURE_OPENAI_ENDPOINT`

### Step 4: Run the Application

```bash
python main.py
```

## 📖 How to Use

### Basic Usage (Interactive Mode)

When you run `python main.py`, you'll see:

1. **Artifact Management Menu:**
   - Choose to keep previous work or start fresh
   - Artifacts = saved progress from previous runs

2. **Task Selection Menu:**
   - Use the default task (NVIDIA stock analysis)
   - Or enter your own custom task
   - Example: "Write a blog about Apple stock performance in Q3 2024"

3. **During Execution:**
   - The AI agents will work together automatically
   - You'll be prompted occasionally to:
     - Press Enter to continue (let agents work)
     - Provide feedback if needed
     - Type 'terminate' to stop early
     - Type 'approved' when satisfied

4. **Final Output:**
   - Find your completed blog post in the `output` folder
   - File will be named like: `report_20241201_143022.md`

### Advanced Usage (Command Line)

```bash
# Run with a custom task directly
python main.py --task "Analyze Tesla stock for the past 3 months"

# Start fresh (clear all previous work)
python main.py --clear-artifacts

# Limit the number of iterations (for faster results)
python main.py --max-inner-turn 5 --max-outer-turn 1

# See all options
python main.py --help
```

## 🤖 The AI Agents Team (agents.py)

The application uses a team of specialized AI agents that collaborate to create your reports. Here's how each agent works:

### 🎩 Admin (User Proxy)
**Role**: You, the human supervisor  
**Responsibilities**:
- Provides the initial task to the team
- Reviews work at each stage
- Gives feedback for improvements
- Approves final results

**How to interact**:
- Press **Enter** to let agents continue working
- Type **feedback** to guide the team
- Type **'approved'** when satisfied
- Type **'terminate'** to stop the process

### 📋 Planner
**Role**: Project manager and quality controller  
**Responsibilities**:
- Analyzes your task requirements
- Delegates work to appropriate agents
- Reviews code execution results
- Checks report quality and SEO
- Ensures factual accuracy

**Key behaviors**:
- **Always** delegates data fetching to Engineer (never tries to fetch data itself)
- Provides specific requirements (date ranges, metrics, chart types)
- Reviews for interactive code issues (plt.show(), input())
- Marks completion with 'REVIEW_COMPLETE' when satisfied

### 💻 Engineer  
**Role**: Technical specialist for data and visualization  
**Responsibilities**:
- Writes Python code to fetch stock data
- Creates data visualizations (charts, graphs)
- Performs calculations and analysis
- Iterates based on feedback

**Technical details**:
- Uses **yfinance** for stock data: `yf.Ticker('NVDA').history(period='1mo')`
- Creates visualizations with **matplotlib**
- Saves data to `coding/data/` folder
- Saves charts to `coding/figures/` folder
- **Never** uses interactive commands (plt.show(), input())
- Always uses non-interactive backend: `matplotlib.use('Agg')`

**Code structure example**:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import yfinance as yf
import matplotlib.pyplot as plt

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Fetch and save data
ticker = yf.Ticker('NVDA')
data = ticker.history(period='1mo')
data.to_csv('data/nvda_data.csv')

# Create and save visualization
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'])
plt.savefig('figures/nvda_price.png')
# NO plt.show() - we save instead!
```

### ⚙️ Executor
**Role**: Code runner and results reporter  
**Responsibilities**:
- Executes Engineer's code
- Reports results and created files
- Shows data statistics
- Provides error messages if code fails

**Safety features**:
- Automatically removes interactive commands
- Replaces `plt.show()` with `plt.savefig()`
- Adds non-interactive backend if missing
- Reports all modifications made

**Execution environment**:
- Working directory: `coding/`
- No Docker required (runs locally)
- Full access to file system for saving

### ✍️ Writer
**Role**: Content creator and SEO specialist  
**Responsibilities**:
- Creates engaging blog posts
- Formats content in Markdown
- Incorporates data and charts
- Optimizes for SEO

**Writing format**:
- Always wraps content in ````md` code blocks
- References charts: `![Chart](figures/chart.png)`
- Uses proper heading hierarchy (H1, H2, H3)
- Includes keywords naturally

**Example output structure**:
````md
# Stock Analysis Title

## Introduction
Engaging opening paragraph...

## Market Performance
![Stock Chart](figures/price_chart.png)
Analysis of the chart...

## Key Metrics
- Metric 1: Value
- Metric 2: Value

## Conclusion
Summary and insights...
````

### 📝 Summarizer
**Role**: Progress tracker and memory keeper  
**Responsibilities**:
- Creates iteration summaries
- Tracks improvements made
- Lists action items
- Saves progress to artifacts

**Output format**:
- JSON-compatible summaries
- Saved to `artifact/` folder
- Includes metrics and progress
- Enables continuation in future sessions

## 🔄 Agent Communication Flow

The agents work together in a specific workflow:

```
1. Admin → provides task → Planner
2. Planner → analyzes needs → Engineer
3. Engineer → writes code → Executor
4. Executor → runs code, reports results → Planner
5. Planner → reviews results → Writer
6. Writer → creates blog post → Planner
7. Planner → reviews content → Admin
8. Admin → provides feedback → (loop or approve)
9. Summarizer → creates iteration summary
```

### Speaker Transitions

The system uses **controlled speaker transitions** to ensure orderly conversation:

| From | Can speak to |
|------|-------------|
| Admin | Planner only |
| Planner | Summarizer, Engineer, Writer |
| Engineer | Executor only |
| Executor | Planner only |
| Writer | Planner only |
| Summarizer | Admin only |

This prevents agents from talking out of turn and ensures logical workflow.

## 🎮 How main.py Works

The `main.py` file is the entry point that orchestrates everything:

### 1. **Directory Management**
- Creates and manages 4 main directories:
  - `output/` - Final reports are saved here
  - `coding/` - Temporary Python code and data files
  - `draft/` - Draft versions of blog posts
  - `artifact/` - Memory and progress from previous runs
- Automatically cleans working directories (coding/draft) each run
- Preserves artifacts unless you choose to clear them

### 2. **Smart Artifact Handling**
- **Preserves Progress**: Keeps artifacts from previous sessions by default
- **Memory System**: Remembers what was done in previous runs
- **Clean Start Option**: Can clear everything with `--clear-artifacts`
- **Auto-Recovery**: If terminated early, extracts the latest draft

### 3. **Iteration Control**
- **Outer Iterations**: Major revision cycles (default: 3)
  - Each outer iteration can refine the entire work
  - Builds on previous iterations' feedback
- **Inner Iterations**: Steps within each outer cycle (default: 100)
  - Individual agent interactions
  - Typically: Plan → Code → Execute → Write → Review

### 4. **Intelligent Task Processing**
```python
# The main workflow in main.py:
1. Setup directories and load configuration
2. Check for existing artifacts (previous work)
3. Initialize memory manager
4. Create AI agents team
5. Run iterations:
   - Outer loop: Major revisions
     - Inner loop: Agent conversations
       - Planner analyzes task
       - Engineer writes code
       - Executor runs code
       - Writer creates content
       - Admin reviews
6. Extract and save final report
```

### 5. **Error Recovery Features**
- **Auto-save drafts**: Every draft is saved with timestamp
- **Termination handling**: Gracefully saves work when stopped
- **Draft extraction**: Finds latest draft if conversation ends early
- **Fallback search**: Checks multiple locations for reports

## 🛠️ Customizing Agent Behavior

You can modify agent behaviors by editing their system messages in `agents.py`:

### Make Engineer use different data sources
Edit the Engineer's system message to include:
```python
"3. Data sources to use:
   - yfinance for stock data
   - pandas_datareader for economic data
   - Alpha Vantage API for detailed metrics"
```

### Change Writer's style
Edit the Writer's system message to specify style:
```python
"2. Writing style:
   - Professional and analytical
   - Include technical indicators
   - Target audience: experienced investors"
```

### Adjust Planner's review criteria
Edit the Planner's system message for different focus:
```python
"6. Review criteria priorities:
   - Technical accuracy (highest)
   - Data completeness
   - Visual appeal
   - SEO optimization"
```

## 🔧 Configuration Options

### Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--task` | Specify task directly | Interactive prompt | `--task "Analyze AAPL stock"` |
| `--max-inner-turn` | Limit agent interactions per outer iteration | 100 | `--max-inner-turn 10` |
| `--max-outer-turn` | Limit major revision cycles | 3 | `--max-outer-turn 2` |
| `--clear-artifacts` | Start fresh, remove all previous work | False | `--clear-artifacts` |
| `--preserve-all` | Debug mode - don't clear any folders | False | `--preserve-all` |
| `--interactive` | Force interactive mode | Auto-detect | `--interactive` |

### Environment Variables for Agent Models

You can assign different AI models to different agents for cost optimization:

```bash
# .env file
AZURE_OPENAI_MODEL=gpt-4              # Default for all agents
AZURE_OPENAI_CODE_WRITER=gpt-4        # Engineer (needs to be smart)
AZURE_OPENAI_CODE_CRITIC=gpt-4        # Planner (needs good judgment)
AZURE_OPENAI_CODE_EXE=gpt-3.5-turbo   # Executor (simpler tasks)
```

## 📊 What Kind of Reports Can It Create?

- Stock performance analysis
- Market trend reports
- Company financial summaries
- Investment opportunity assessments
- Comparative stock analyses
- Sector performance reviews

## 🛡️ Security & Privacy

- **API Keys:** Keep your `.env` file private - never share or commit to git
- **Data:** Stock data is fetched from public Yahoo Finance API
- **Storage:** All generated content stays on your local machine

## 🆘 Getting Help

1. **Check the logs:** Error messages usually explain what went wrong
2. **Review output folder:** Previous successful reports can guide you
3. **Reduce complexity:** Start with simple tasks, then get more complex
4. **Check API limits:** Ensure your Azure account has sufficient quota

## 🎯 Tips for Best Results

1. **Be Specific:** The more detailed your task, the better the output
2. **Let It Run:** Don't terminate early - agents improve with iterations
3. **Provide Feedback:** When prompted, guide the agents toward what you want
4. **Save Artifacts:** Keep artifacts to build upon previous work
5. **Check Charts:** Review the `coding/figures/` folder for generated charts
6. **Understanding Agent Feedback**: 
   - If Planner requests changes, it's improving quality
   - Multiple iterations are normal and beneficial
   - Each agent specializes in their domain

7. **Debugging Issues**:
   - Check `coding/` folder for generated Python files
   - Review `draft/` folder for report iterations
   - Look for error messages from Executor

## 📈 Understanding the Output

Your final report will include:
- **Headlines & Sections:** Properly formatted with markdown
- **Data & Statistics:** Real numbers from actual stock data
- **Charts & Visualizations:** Embedded images showing trends
- **Analysis:** AI-generated insights based on the data
- **SEO Elements:** Keywords and structure for web publishing

The report is in Markdown format (`.md` file) which can be:
- Opened in any text editor
- Converted to HTML for websites
- Imported into blogging platforms
- Converted to PDF for reports

## 🔄 Updating the Application

```bash
# Get latest updates
git pull

# Update dependencies
pip install -r requirements.txt --upgrade
```

## 📜 License

[Your License Here]

## 🙏 Credits

Built with:
- [AG2 (AutoGen)](https://github.com/ag2ai/ag2) - Multi-agent conversation framework
- Azure OpenAI - AI language models
- Yahoo Finance - Stock data API
- Python - Programming language

---

**Need more help?** Create an issue on GitHub or contact the maintainers.

**Enjoy creating professional stock reports with AI! 🚀**
