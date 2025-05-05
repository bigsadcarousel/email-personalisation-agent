# Personalization Agent (Beta V0) ✨📧

Personalization Agent is a Streamlit web application designed to generate personalized cold email opening lines based on the content of a provided web page URL (e.g., a company's 'About Us' page, a person's LinkedIn profile, a blog post).

## Purpose

The primary goals of this Beta are:

*   **Demonstrate Capability:** Showcase the ability to generate relevant, personalized opening lines by analyzing web content.
*   **Test Core Technology:** Evaluate the effectiveness of web scraping (Firecrawl) combined with configurable LLMs (OpenAI via Agents SDK) for automated copywriting assistance.
*   **Gather Usage Data:** Collect basic metrics on user interactions (URLs used, email purpose, generated lines) to inform future development and prompt refinement.

## Features

*   **Single URL Intake:** Accepts a single web page URL as input.
*   **Web Content Extraction:** Uses the Firecrawl API (`firecrawl-py` SDK) to scrape and convert the specified web page content into clean Markdown text. Includes an option for potentially longer wait times to handle JavaScript-rendered content.
*   **Configurable AI Model:** Allows users to select from a list of OpenAI models (e.g., `gpt-4o`, `gpt-4o-mini`) via the OpenAI Agents SDK to perform the generation task.
*   **Optional Email Purpose:** Users can provide context about the purpose of their email (e.g., "Sales pitch for SEO services", "Job application for marketing role") to help guide the generation.
*   **Personalized Opening Line Generation:** The selected AI agent analyzes the scraped text and the optional email purpose to generate *one* concise opening sentence. The underlying prompt prioritizes using recent achievements, specific testimonial outcomes, or relevant article topics found in the text.
*   **Usage Logging:** Records timestamp, source URL, email purpose, and the generated opening line to `usage_log.csv` for analysis.
*   **Basic Input Validation:** Checks for reasonable URL length/format.

## Tech Stack

*   **Language:** Python 3.x
*   **Web Framework/UI:** Streamlit
*   **AI Agent Orchestration:** OpenAI Agents SDK (`openai-agents`, `openai`)
*   **Web Scraping:** Firecrawl (`firecrawl-py`)
*   **Text Processing:** Tiktoken (for chunking text based on token counts)
*   **Configuration:** python-dotenv (for managing API keys via `.env` file locally)
*   **Data Storage (Local/MVP):** CSV (for usage logs)

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
    .\.venv\Scripts\activate.bat
    # Activate (Windows - PowerShell)
    .\.venv\Scripts\Activate.ps1 
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
    *   The deployed app will read these secrets as environment variables.

## Data Files

The application generates the following file locally or within the deployed container:

*   `usage_log.csv`: Appends records of URLs processed, email purpose provided, and the generated opening line.

This file is listed in `.gitignore` and should not be committed to version control. When deployed on Streamlit Community Cloud, this file resides within the app's running container. You may need to download it periodically from the cloud interface if you need to analyze the data, as container filesystems might not be permanently persistent across all updates or restarts. For more robust data handling, consider integrating a database.
