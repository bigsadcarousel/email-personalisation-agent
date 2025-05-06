import os, json, uuid, textwrap, asyncio, csv
from pathlib import Path
from typing import List
from datetime import datetime
import pandas as pd

import streamlit as st
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI
from agents import Agent, Runner
import tiktoken

# ----------------- ENV & CONSTANTS -----------------
load_dotenv()

APP_NAME    = "Personalization Agent (Beta V0)"
DATA_DIR    = Path(__file__).parent
FEEDBACK_FILE = DATA_DIR / "feedback.csv"
USAGE_LOG_FILE = DATA_DIR / "usage_log.csv"

OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
FIRECRAWL_KEY   = os.getenv("FIRECRAWL_API_KEY")

MAX_URL_LENGTH = 1000

if not (OPENAI_KEY and FIRECRAWL_KEY):
    st.stop("❌ Set OPENAI_API_KEY and FIRECRAWL_API_KEY in a .env file.")

firecrawl = FirecrawlApp(api_key=FIRECRAWL_KEY)
client    = OpenAI(api_key=OPENAI_KEY)

# Initialize session state
if 'generated_line' not in st.session_state:
    st.session_state.generated_line = None
if 'page_text' not in st.session_state:
    st.session_state.page_text = None
if 'source_url' not in st.session_state:
    st.session_state.source_url = None
if 'csv_results' not in st.session_state:
    st.session_state.csv_results = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None

# ----------------- HELPERS -----------------

# Improved Chunking Function
def create_context_chunks(markdown_text: str, max_tokens=3000, overlap=100) -> List[str]:
    """Creates text chunks respecting paragraphs, splitting large paragraphs with overlap."""
    enc = tiktoken.encoding_for_model("gpt-4o-mini") # Align with agent model if different

    paragraphs = markdown_text.split('\n\n')

    chunks = []
    current_chunk_para_list = []
    current_token_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = enc.encode(para)
        para_token_count = len(para_tokens)

        # Case 1: Single paragraph is too large
        if para_token_count > max_tokens:
            # If there's content in the current chunk, add it to chunks list first
            if current_chunk_para_list:
                chunks.append('\n\n'.join(current_chunk_para_list))
                current_chunk_para_list = []
                current_token_count = 0

            # Split the large paragraph with overlap
            start_index = 0
            while start_index < para_token_count:
                end_index = min(start_index + max_tokens, para_token_count)
                chunk_tokens = para_tokens[start_index:end_index]
                chunks.append(enc.decode(chunk_tokens))
                # Move start index for the next chunk, considering overlap
                start_index += (max_tokens - overlap)
                if start_index >= end_index: # Prevent infinite loop on very small overlap/max_tokens
                    break
            continue # Move to the next paragraph

        # Case 2: Adding this paragraph would exceed the limit
        # Add 2 tokens for the potential '\n\n' separator
        if current_token_count + para_token_count + (2 if current_chunk_para_list else 0) > max_tokens:
            # Add the current chunk to the list
            if current_chunk_para_list:
                chunks.append('\n\n'.join(current_chunk_para_list))

            # Start new chunk with the current paragraph
            current_chunk_para_list = [para]
            current_token_count = para_token_count
        # Case 3: Adding this paragraph fits
        else:
            current_chunk_para_list.append(para)
            # Add token count for paragraph and potentially the separator
            current_token_count += para_token_count + (2 if len(current_chunk_para_list) > 1 else 0)

    # Add any remaining content in the last chunk
    if current_chunk_para_list:
        chunks.append('\n\n'.join(current_chunk_para_list))

    # If no chunks were created (e.g., empty input), return a list with an empty string
    if not chunks:
         return [""]

    return chunks

def scrape_page(url: str, wait_ms: int = 2000) -> str:
    """Scrapes a single URL using Firecrawl, returning Markdown content."""
    # Potential future improvements:
    # - Explore `pageOptions={'onlyMainContent': True}` if switching to JSON format.
    # - Use Firecrawl 'actions' for pages requiring complex interaction.
    # - More specific error handling for Firecrawl exceptions.
    md_content = ""
    try:
        print(f"Attempting to scrape: {url} with wait {wait_ms}ms") # Debug print
        res = firecrawl.scrape_url(url, formats=["markdown"], waitFor=wait_ms)
        # print(f"Scrape response received: {res}") # Debug print
        md_content = res.markdown if res else ""
        if not md_content:
            print(f"Scraping resulted in empty content for: {url}") # Debug print
            # More specific error message
            raise RuntimeError(f"🔥 No Markdown text extracted from {url}. Page might block scraping or need longer wait.")
        print(f"Scraping successful for: {url}") # Debug print
        return md_content # Return successfully scraped content
    except Exception as e:
        print(f"Scraping failed for {url}. Error: {e}") # Debug print
        # Catch potential Firecrawl specific errors if the SDK defines them, otherwise generic
        raise RuntimeError(f"Scraping failed for {url}. Reason: {e}")

# Refined Agent Prompt V11 (Include Relevant Testimonials/Articles)
AGENT_PROMPT = textwrap.dedent("""\
    You are an elite AI copywriter crafting personalized cold‑email opening lines for Remotebase.   
    Remotebase helps tech leaders hire pre‑vetted remote engineers fast, cut hiring cycles, and scale teams cost‑effectively.

    INPUT FORMAT:
    - PAGE_URL: The URL where the text originated.  
    - PAGE_TEXT: The content scraped from that page.  
    - EMAIL_PURPOSE: (Optional) The specific Remotebase offer or angle, e.g., "fill open senior‑backend roles quickly," "cover a skills gap before funding closes," etc.

    CRITICAL CONSTRAINTS:
    ● **USE ONLY PROVIDED TEXT:** Base the line strictly on the PAGE_TEXT—no outside facts or guesses.  
    ● **NO INVENTION:** Do not invent numbers, achievements, pain points, or dates.  
    ● **ONE SENTENCE ONLY:** Produce exactly one concise sentence (≤ 30 words).  
    ● **FINAL FALLBACK:** If nothing usable exists, output the exact phrase:  
      `No usable opening line found based on the provided text.`

    HIERARCHY & STEPS (stop at the first that succeeds):

    1. **Specific Pain/Trigger (Priority 1):**  
       • Hunt PAGE_TEXT for clear signals that the company is hiring engineers, expanding product lines, recently funded, facing talent shortages, or experiencing rapid growth.  
       • If found **and** relevant to EMAIL_PURPOSE, craft a natural sentence linking that trigger to how Remotebase supplies vetted engineers fast (mentioning time‑to‑hire, quality, or cost benefits).  

    2. **Recent Achievement (Priority 2):**  
       • If no hiring trigger, look for verifiable, recent wins: funding rounds, product launches, user milestones, speed‑to‑market claims, engineering accolades.  
       • Tie that win to how Remotebase can help sustain or accelerate momentum with on‑demand engineering talent.  

    3. **Broader Alignment (Priority 3):**  
       • If nothing above, find a broad technology‑stack or remote‑work reference (e.g., "building in React," "remote‑first culture").  
       • Write a sentence linking that reference to Remotebase's remote developer network.  

    4. **Generic Compliment (Priority 4):**  
       • If still nothing, identify the company or person's name and write a polite, generic acknowledgment of their work, segueing to Remotebase's value.  

    5. **Final Fallback (Priority 5):**  
       • If no usable info at all, output: `No usable opening line found based on the provided text.`

    OUTPUT RULES:
    – Output **only** the single sentence or the fallback phrase—no labels, quotes, or extra commentary.  
    – Keep wording varied and natural across different runs.
""")

# Define the Personalization Agent globally - COMMENTING OUT
# personalization_agent = Agent(
#     name="Personalization Agent",
#     instructions=AGENT_PROMPT,
#     model="o4-mini" # Keeping gpt-4o
# )

# Renamed function to run the Personalization Agent
async def run_personalization_agent(agent: Agent, input_text: str) -> str:
    """Runs the personalization agent with the provided input text."""
    try:
        result = await Runner.run(agent, input_text)
        return result.final_output
    except Exception as e:
        st.error(f"Error running agent: {e}")
        return "Sorry, I encountered an error trying to generate the opening line."

# ----------------- STREAMLIT LAYOUT -----------------
st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="wide")

# Custom theme colors
st.markdown("""
<style>
    .main-header {color: #8661c5;}
    .subheader {color: #6c7ac9;}
    .status-msg {background-color: #2d3047; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("1. Input Method")
    input_method = st.radio(
        "Choose input method:",
        ["Single URL", "CSV Upload"],
        index=0
    )

    if input_method == "Single URL":
        st.header("2. Analyze Page")
        url = st.text_input("Enter profile/company URL...", placeholder="https://example.com/about")
        use_wait = st.checkbox("Use comprehensive text extraction (wait for JS)")
        analyze_button = st.button("📊 Analyze Page Content", use_container_width=True)
    else:
        st.header("2. Upload CSV")
        uploaded_file = st.file_uploader("Upload CSV file with URLs", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of your CSV:")
            st.dataframe(df.head())
            url_column = st.selectbox("Select the column containing URLs", df.columns)
            if url_column:
                st.session_state.csv_urls = df[url_column].tolist()
                st.write(f"Found {len(st.session_state.csv_urls)} URLs to process")

    st.divider()
    
    st.header("3. Generate Lines")
    email_purpose = st.text_input("Purpose of email (optional context):", placeholder="e.g., Job application, Sales pitch")

    # Model Selection
    model_options = {
        "GPT-4o": "gpt-4o",
        "GPT-4o-mini": "gpt-4o-mini",
        "o4-mini": "o4-mini",
        "o3": "o3",
        "o3-mini": "o3-mini",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
    }
    selected_model_name = st.selectbox(
        "Select Generation Model:",
        options=list(model_options.keys()),
        index=1
    )

    # Generate button is enabled based on input method
    if input_method == "Single URL":
        generate_button = st.button("✨ Generate Opening Line", 
                                  use_container_width=True, 
                                  disabled=not st.session_state.get("page_text"))
    else:
        generate_button = st.button("✨ Process CSV", 
                                  use_container_width=True, 
                                  disabled=not st.session_state.get("csv_urls"))

# ---- Main pane ----
st.title(f"✨ {APP_NAME}")
st.markdown('<p class="subheader">Generate personalized cold email opening lines from web content.</p>', unsafe_allow_html=True)

# --- Stage 1: Analysis Logic ---
if input_method == "Single URL" and analyze_button:
    # Reset state before analysis
    st.session_state.page_text = None
    st.session_state.source_url = None
    st.session_state.generated_line = None

    # Validate URL
    if not url:
        st.warning("Please enter a URL.")
    elif len(url) > MAX_URL_LENGTH:
        st.error(f"URL is too long (max {MAX_URL_LENGTH} characters).")
    elif "://" not in url:
        st.error("Please enter a valid URL (including http:// or https://).")
    else:
        # Scrape Page
        try:
            scrape_wait_time = 4000 if use_wait else 2000
            with st.spinner("Reading page... (please wait)"):
                page_md = scrape_page(url, wait_ms=scrape_wait_time)
            # Store successful scrape results in session state
            st.session_state.page_text = page_md
            st.session_state.source_url = url
            st.success(f"✅ Analyzed content loaded for: {url}")
            st.rerun()
        except Exception as e:
            st.error(f"Could not analyze page: {e}")
            st.session_state.page_text = None
            st.session_state.source_url = None
            st.session_state.generated_line = None

# --- Stage 2: Generation Logic --- 
if generate_button:
    if input_method == "Single URL":
        # Single URL processing
        if st.session_state.get("page_text") and st.session_state.get("source_url"):
            with st.spinner("Generating opening line... (please wait)"):
                try:
                    selected_model_id = model_options[selected_model_name]
                    local_personalization_agent = Agent(
                        name="Personalization Agent",
                        instructions=AGENT_PROMPT,
                        model=selected_model_id
                    )
                    
                    context_chunks = create_context_chunks(st.session_state.page_text, max_tokens=3000)
                    full_context = "\n\n---\n\n".join(context_chunks)[:80000]
                    
                    purpose_string = f"EMAIL_PURPOSE: {email_purpose}\n\n" if email_purpose else ""
                    input_text = (
                        f"PAGE_URL: {st.session_state.source_url}\n\n"
                        f"{purpose_string}"
                        f"PAGE_TEXT:\n{full_context}"
                    )
                    
                    generated_line = asyncio.run(run_personalization_agent(local_personalization_agent, input_text))
                    st.session_state.generated_line = generated_line
                    
                    # Log usage
                    try:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_data = [timestamp, st.session_state.source_url, email_purpose, generated_line]
                        DATA_DIR.mkdir(parents=True, exist_ok=True)
                        with open(USAGE_LOG_FILE, "a", newline='', encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(log_data)
                    except Exception as log_e:
                        st.warning(f"Could not log usage data: {log_e}")

                except Exception as agent_e:
                    st.error(f"Generation error: {agent_e}")
                    st.session_state.generated_line = "Error generating line."
    else:
        # CSV processing
        if st.session_state.get("csv_urls"):
            st.session_state.processing_status = "Processing URLs..."
            st.session_state.csv_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, url in enumerate(st.session_state.csv_urls):
                try:
                    status_text.text(f"Processing {i+1}/{len(st.session_state.csv_urls)}: {url}")
                    progress_bar.progress((i + 1) / len(st.session_state.csv_urls))
                    
                    # Scrape page
                    page_md = scrape_page(url, wait_ms=4000)
                    
                    # Create context chunks
                    context_chunks = create_context_chunks(page_md, max_tokens=3000)
                    full_context = "\n\n---\n\n".join(context_chunks)[:80000]
                    
                    # Prepare input for the agent
                    purpose_string = f"EMAIL_PURPOSE: {email_purpose}\n\n" if email_purpose else ""
                    input_text = (
                        f"PAGE_URL: {url}\n\n"
                        f"{purpose_string}"
                        f"PAGE_TEXT:\n{full_context}"
                    )
                    
                    # Create and run the agent
                    selected_model_id = model_options[selected_model_name]
                    agent = Agent(
                        name="Personalization Agent",
                        instructions=AGENT_PROMPT,
                        model=selected_model_id
                    )
                    
                    result = asyncio.run(run_personalization_agent(agent, input_text))
                    
                    st.session_state.csv_results.append({
                        "url": url,
                        "opening_line": result,
                        "status": "success"
                    })
                    
                except Exception as e:
                    st.session_state.csv_results.append({
                        "url": url,
                        "opening_line": f"Error: {str(e)}",
                        "status": "error"
                    })
            
            # Convert results to DataFrame and display
            results_df = pd.DataFrame(st.session_state.csv_results)
            st.dataframe(results_df)
            
            # Add download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="opening_lines_results.csv",
                mime="text/csv"
            )

# --- Display Area (runs on every interaction) --- 

# Display info about the loaded page (for single URL mode)
if input_method == "Single URL" and st.session_state.get("source_url"):
    st.divider()
    st.info(f"Analyzed content loaded for: {st.session_state.source_url}")
    
    if st.session_state.get("page_text"):
         with st.expander("View Analyzed Page Content (Text Format)", expanded=False):
            st.text_area("Page Text:", value=st.session_state.page_text,
                        height=300, key="analyzed_text_view", disabled=True)

# Display the generated line if available (for single URL mode)
if input_method == "Single URL" and st.session_state.get("generated_line"):
    st.subheader("✨ Suggested Opening Line:")
    st.text_area("Copy this line:", value=st.session_state.generated_line, height=100, key="output_line")
elif input_method == "Single URL" and generate_button and not st.session_state.get("generated_line"):
     st.warning("Could not generate opening line. Check for errors above or try again.")

# Initial message if no analysis has been triggered yet
if input_method == "Single URL" and not st.session_state.get("page_text") and not analyze_button:
    st.info("👈 Enter a URL in the sidebar and click **Analyze Page Content** to start.")