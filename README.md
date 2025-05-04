# Uni-Agent (Beta V0) 🤖🎓

Uni-Agent is a Streamlit web application designed as a Minimum Viable Product (MVP) to serve as an AI assistant for prospective international students. Users can input the URL of a specific university web page (e.g., admissions requirements, deadlines, program details), and the agent will answer questions based *only* on the content of that page, providing cited and structured responses.

## Purpose

The primary goals of this MVP are:

*   **Validate Demand:** Quickly gauge user interest in an interactive Q&A interface specifically tailored to the content of individual university web pages, as an alternative to navigating complex websites.
*   **Test Core Technology:** Evaluate the effectiveness of web scraping (Firecrawl) combined with LLM question-answering (OpenAI Agents SDK) for this specific use case.
*   **Capture Early Leads:** Offer an optional email signup for users interested in future updates and more advanced features.
*   **Gather Usage Data:** Collect basic metrics on user interactions (URLs used, questions asked, answers generated) to inform future development.

## Features

*   **Single URL Intake:** Accepts a single university web page URL as input.
*   **Web Content Extraction:** Uses the Firecrawl API (`firecrawl-py` SDK) to scrape and convert the specified web page content into clean Markdown text. Includes options for handling JavaScript-rendered content via wait times.
*   **AI-Powered Q&A:** Leverages the OpenAI Agents SDK (`gpt-4.1` or similar model specified) to answer user questions based *strictly* on the extracted page text.
*   **Conversational Context:** Remembers the last few turns of the conversation within a session to understand follow-up questions.
*   **Structured & Cited Answers:** The agent is prompted to provide concise answers, use bullet points for lists (max 5), and cite the source URL once at the beginning of its response.
*   **Usage Limits:**
    *   **Session Limit:** Restricts users to a specific number of questions (e.g., 5) per analyzed page within a single browser session.
    *   **Persistent Global Limit:** A configurable overall limit (`RUN_LIMIT`) on the total number of questions the deployed application instance can answer across all users, tracked via `limit.json`.
*   **Optional Email Signup:** A sidebar form allows users to optionally subscribe for updates, saving emails to `emails.csv`.
*   **Usage Logging:** Records timestamp, source URL, user question, and agent answer to `usage_log.csv` for analysis.
*   **Feedback Collection:** An optional in-app form allows users to submit feedback, saved to `feedback.csv`.
*   **Basic Input Validation:** Checks for reasonable URL length/format and question length.

## Tech Stack

*   **Language:** Python 3.x
*   **Web Framework/UI:** Streamlit
*   **AI Agent Orchestration:** OpenAI Agents SDK (`openai-agents`, `openai`)
*   **Web Scraping:** Firecrawl (`firecrawl-py`)
*   **Text Processing:** Tiktoken (for chunking text based on token counts)
*   **Configuration:** python-dotenv (for managing API keys via `.env` file locally)
*   **Data Storage (Local/MVP):** JSON (for run limit), CSV (for emails, feedback, usage logs)

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd study-abroad-agent
    ```
2.  **Create Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv 
    # Activate (Linux/macOS)
    source .venv/bin/activate
    # Activate (Windows - Git Bash/WSL)
    source .venv/Scripts/activate
    # Activate (Windows - Command Prompt)
    .\\.venv\\Scripts\\activate.bat
    # Activate (Windows - PowerShell)
    .\\.venv\\Scripts\\Activate.ps1 
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create `.env` File:** Create a file named `.env` in the root directory (`study-abroad-agent/`).
5.  **Add API Keys:** Open the `.env` file and add your API keys:
    ```dotenv
    OPENAI_API_KEY="sk-..."
    FIRECRAWL_API_KEY="fc-..."
    # Optional: Set a global run limit (defaults to 100 if not set)
    # RUN_LIMIT=500 
    ```
    *Replace `sk-...` and `fc-...` with your actual keys.*

## Running Locally

1.  Ensure your virtual environment is activated.
2.  Navigate to the project directory (`study-abroad-agent/`).
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  Streamlit will provide a local URL (usually `http://localhost:8501`) to access the app in your browser.

## Deployment (Streamlit Community Cloud)

Streamlit Community Cloud is a fast way to deploy this app publicly for free.

1.  **Push to GitHub:** Make sure your code, including `app.py`, `requirements.txt`, and the `.gitignore` file, is pushed to a GitHub repository. **Do NOT commit your `.env` file.**
2.  **Sign Up/In:** Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3.  **Deploy App:**
    *   Click "New app" -> "From repo".
    *   Select your repository, branch (e.g., `main`), and the main application file (`app.py`).
    *   Click "Deploy!".
4.  **Add Secrets:** This is crucial for security. In your deployed app's settings on Streamlit Community Cloud (usually under the menu ☰ -> Settings -> Secrets):
    *   Add `OPENAI_API_KEY` and paste your OpenAI key as the value.
    *   Add `FIRECRAWL_API_KEY` and paste your Firecrawl key as the value.
    *   Optionally, add `RUN_LIMIT` if you want to set a global limit different from the default 100.
    *   The deployed app will read these secrets as environment variables.

## Data Files

The application generates the following files locally or within the deployed container:

*   `limit.json`: Stores the count for the persistent global `RUN_LIMIT`.
*   `emails.csv`: Appends email addresses from the optional signup form.
*   `feedback.csv`: Appends user feedback submitted through the app.
*   `usage_log.csv`: Appends records of URLs processed, questions asked, and answers given.

These files are listed in `.gitignore` and should not be committed to version control. When deployed on Streamlit Community Cloud, these files reside within the app's running container. You may need to download them periodically from the cloud interface if you need to analyze the data, as container filesystems might not be permanently persistent across all updates or restarts. For more robust data handling, consider integrating a database.
