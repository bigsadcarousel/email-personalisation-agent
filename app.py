import os, json, uuid, textwrap, asyncio, csv
from pathlib import Path
from typing import List
from datetime import datetime

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
    You are an expert AI copywriter creating personalized cold email opening lines. Your goal is to generate ONE concise sentence based *exclusively* on the provided web page text, following a hierarchy of personalization. Prioritize recent, relevant achievements, testimonials, or articles.

    INPUT FORMAT:
    - PAGE_URL: The URL where the text originated.
    - PAGE_TEXT: The content scraped from the page.
    - EMAIL_PURPOSE: (Optional) The reason for sending the email.

    CRITICAL CONSTRAINTS:
    - **USE ONLY PROVIDED TEXT:** Base your sentence *strictly* on information found in the PAGE_TEXT. Don't infer dates unless explicitly stated.
    - **NO MAKING STUFF UP:** Absolutely do not invent facts, infer details, or use any external knowledge.
    - **FINAL FALLBACK:** If you cannot find *any* usable information to generate *any* kind of opening line, output *only* this exact phrase: `No usable opening line found based on the provided text.`

    YOUR TASK (Follow in order):
    1.  **Understand Purpose & Subject:** Note the `EMAIL_PURPOSE`. Identify the main subject (person/company).
    2.  **Attempt Specific Connection (Priority 1 - Focus on Recent/Relevant Content):**
        *   Scan `PAGE_TEXT` for *specific, verifiable* details. Prioritize:
            *   Recent achievements or announcements (e.g., within the last ~12 months).
            *   **Key positive outcomes or specific benefits highlighted within testimonials.** (Quote the outcome/benefit concisely if needed).
            *   **Titles or topics of relevant articles/blog posts mentioned.**
        *   Also look for other specifics: unique mission phrases, specific projects/roles, awards, etc.
        *   Assess if a found detail **directly and clearly connects** to the `EMAIL_PURPOSE`. **Strongly prefer using recent/relevant achievements, testimonial outcomes, or article topics if they connect well.**
        *   **If** such a specific, relevant detail is found: Generate a concise (max 25-30 words), natural sentence linking this specific detail (achievement, testimonial outcome, article topic) to the purpose. Ensure varied phrasing. **STOP HERE and output this sentence.**
    3.  **Attempt Broad Connection (Priority 2):**
        *   **If no direct connection using Priority 1 details:** Look for a specific detail (service, general expertise area) in the `PAGE_TEXT` that falls into a *broader category* relevant to the `EMAIL_PURPOSE`.
        *   **If found:** Generate a concise sentence linking the purpose to this broader category. Acknowledge the connection might be general. **STOP HERE and output this sentence.**
    4.  **Generate Generic Personalized Opener (Priority 3):**
        *   **If no specific or broad connection possible:** Identify the primary company or person's name.
        *   **If yes:** Generate a polite, generic opening line (max 20 words) acknowledging their work/website, using their name. **STOP HERE and output this sentence.**
    5.  **Use Final Fallback (Priority 4):**
        *   **If you cannot even generate a generic opener:** Output the final fallback phrase: `No usable opening line found based on the provided text.`

    OUTPUT RULES:
    - Output ONLY the single sentence generated by the first successful step OR the final fallback phrase.
    - No extra text, labels, explanations, or quotes.

    Example 1 (Using Relevant Testimonial Outcome - Step 2):
    PAGE_URL: https://example.com/testimonials
    PAGE_TEXT: "...'Their tool cut our reporting time in half!' - Jane D, CEO..."
    EMAIL_PURPOSE: Sell an advanced reporting tool focusing on time savings.
    Example Output 1: Learning how you helped Jane D. cut reporting time in half really highlights the efficiency gains possible, similar to our advanced tool.

    Example 2 (Using Relevant Article Topic - Step 2):
    PAGE_URL: https://example.com/blog
    PAGE_TEXT: "...Our latest article discusses 'Integrating AI for Sustainable Manufacturing'..."
    EMAIL_PURPOSE: Offer partnership on sustainable AI monitoring.
    Example Output 2: Your recent article on 'Integrating AI for Sustainable Manufacturing' caught my eye, as it aligns closely with our AI monitoring solutions.

    Example 3 (Specific Recent Achievement - Step 2):
    PAGE_URL: https://example.com/news/q3-results
    PAGE_TEXT: "...achieved 150% growth in user adoption for our new WidgetPro tool in Q3..."
    EMAIL_PURPOSE: Partnership proposal related to enhancing WidgetPro adoption further.
    Example Output 3: That 150% Q3 user adoption growth for WidgetPro is impressive – our platform specializes in accelerating exactly that engagement.

    Example 4 (Specific Older Detail - Step 2):
    PAGE_URL: https://www.dbfwclegal.com/jim-y-wong/
    PAGE_TEXT: "...His business law practice encompasses...visa applications and immigration compliance..." (Assume no relevant recent achievement found)
    EMAIL_PURPOSE: Introduce AI platform for O1/EB1A immigration workflows.
    Example Output 4: Noting your firm's work in visa applications and immigration compliance, our AI platform specifically designed for O1/EB1A workflows could be valuable.

    Example 5 (Broad Connection - Step 3):
    PAGE_URL: https://champlawgroup.com/
    PAGE_TEXT: "...Handling Employment Immigration matters..." (Assume no relevant recent achievement or direct match)
    EMAIL_PURPOSE: Introduce AI platform for O1/EB1A immigration workflows.
    Example Output 5: Seeing that Champagne Law Group handles Employment Immigration, our AI platform for specific visa types like O1/EB1A might align with your practice.

    Example 6 (Generic Personalized - Step 4):
    PAGE_URL: https://coolwidgets.com/
    PAGE_TEXT: (Minimal text)
    EMAIL_PURPOSE: General sales outreach.
    Example Output 6: Came across CoolWidgets' website and wanted to introduce our services.

    Example 7 (Final Fallback - Step 5):
    PAGE_URL: https://example.com/blank
    PAGE_TEXT: ""
    EMAIL_PURPOSE: Sales outreach
    Example Output 7: No usable opening line found based on the provided text.
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
st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="wide") # Changed icon

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
    st.header("1. Analyze Page")
    url = st.text_input("Enter profile/company URL...", placeholder="https://example.com/about")
    use_wait = st.checkbox("Use comprehensive text extraction (wait for JS)")
    analyze_button = st.button("📊 Analyze Page Content", use_container_width=True)

    st.divider()
    
    st.header("2. Generate Line")
    email_purpose = st.text_input("Purpose of email (optional context):", placeholder="e.g., Job application, Sales pitch")

    # Model Selection
    model_options = {
        # Standard OpenAI Models
        "GPT-4o": "gpt-4o",
        "GPT-4o-mini": "gpt-4o-mini",

        # Aliases / Non-Standard IDs (May require specific setup/library support)
        "o4-mini": "o4-mini",
        "o3": "o3",
        "o3-mini": "o3-mini",

        # Azure/Experimental Models (May require specific setup/library support)
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
    }
    selected_model_name = st.selectbox(
        "Select Generation Model:",
        options=list(model_options.keys()),
        index=1 # Default to the standard "GPT-4o Mini"
    )

    # Generate button is enabled only if page_text exists in session state
    generate_button = st.button("✨ Generate Opening Line", 
                              use_container_width=True, 
                              disabled=not st.session_state.get("page_text"))

# ---- Main pane ----
st.title(f"✨ {APP_NAME}")
st.markdown('<p class="subheader">Generate personalized cold email opening lines from web content.</p>', unsafe_allow_html=True)

# --- Stage 1: Analysis Logic ---
if analyze_button:
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
            # Use rerun to update button state immediately after analysis success
            st.rerun() 
        except Exception as e:
            st.error(f"Could not analyze page: {e}")
            # Ensure state is clear on failure
            st.session_state.page_text = None
            st.session_state.source_url = None
            st.session_state.generated_line = None

# --- Stage 2: Generation Logic --- 
if generate_button:
    # Ensure page_text is loaded (should be guaranteed by button state, but check again)
    if st.session_state.get("page_text") and st.session_state.get("source_url"):
        with st.spinner("Generating opening line... (please wait)"):
            try:
                # --- Agent Creation moved inside button click ---
                # Retrieve the selected model ID
                selected_model_id = model_options[selected_model_name]

                # Dynamically create the agent with the selected model
                local_personalization_agent = Agent(
                    name="Personalization Agent",
                    instructions=AGENT_PROMPT, # Reuse the existing prompt
                    model=selected_model_id # Use the ID selected by the user
                )
                # --- End Agent Creation ---

                context_chunks = create_context_chunks(st.session_state.page_text, max_tokens=3000)
                full_context = "\n\n---\n\n".join(context_chunks)[:80000]
                
                purpose_string = ""
                # Retrieve purpose from the input field *at the time of clicking generate*
                current_email_purpose = email_purpose 
                if current_email_purpose:
                    purpose_string = f"EMAIL_PURPOSE: {current_email_purpose}\n\n"

                input_text = (
                    f"PAGE_URL: {st.session_state.source_url}\n\n"
                    f"{purpose_string}"
                    f"PAGE_TEXT:\n{full_context}"
                )
                
                # Run the dynamically configured agent
                generated_line = asyncio.run(run_personalization_agent(local_personalization_agent, input_text))
                st.session_state.generated_line = generated_line
                
                # Log usage
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_data = [ timestamp, st.session_state.source_url, current_email_purpose, generated_line ]
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
        # This case should ideally not happen if button is disabled correctly
        st.warning("Please analyze a page first before generating a line.")

# --- Display Area (runs on every interaction) --- 

# Display info about the loaded page
if st.session_state.get("source_url"):
    st.divider()
    st.info(f"Analyzed content loaded for: {st.session_state.source_url}")
    
    # Add expander for scraped text here for context
    if st.session_state.get("page_text"):
         with st.expander("View Analyzed Page Content (Text Format)", expanded=False):
            st.text_area("Page Text:", value=st.session_state.page_text,
                        height=300, key="analyzed_text_view", disabled=True)

# Display the generated line if available
if st.session_state.get("generated_line"):
    st.subheader("✨ Suggested Opening Line:")
    st.text_area("Copy this line:", value=st.session_state.generated_line, height=100, key="output_line")
# Display warning if generation was attempted but failed (and not due to limits)
elif generate_button and not st.session_state.get("generated_line"):
     st.warning("Could not generate opening line. Check for errors above or try again.")

# Initial message if no analysis has been triggered yet
if not st.session_state.get("page_text") and not analyze_button:
    st.info("👈 Enter a URL in the sidebar and click **Analyze Page Content** to start.")