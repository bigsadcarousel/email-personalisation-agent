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
LIMIT_FILE  = DATA_DIR / "limit.json"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"
USAGE_LOG_FILE = DATA_DIR / "usage_log.csv"

OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
FIRECRAWL_KEY   = os.getenv("FIRECRAWL_API_KEY")
RUN_LIMIT       = int(os.getenv("RUN_LIMIT", "100")) # Persistent global limit

# Renamed/Adjusted Limits
SESSION_GENERATION_LIMIT = 5 # Max generations per analyzed URL in a session
MAX_URL_LENGTH = 1000
# MAX_QUESTION_LENGTH removed - no longer applicable

if not (OPENAI_KEY and FIRECRAWL_KEY):
    st.stop("❌ Set OPENAI_API_KEY and FIRECRAWL_API_KEY in a .env file.")

firecrawl = FirecrawlApp(api_key=FIRECRAWL_KEY)
client    = OpenAI(api_key=OPENAI_KEY)

# Initialize session state
if 'session_runs' not in st.session_state: # Renamed conceptually to session_generations
    st.session_state.session_runs = 0
if 'generated_line' not in st.session_state:
    st.session_state.generated_line = None
if 'page_text' not in st.session_state: # Keep page_text for context
    st.session_state.page_text = None
if 'source_url' not in st.session_state:
    st.session_state.source_url = None
# 'messages' session state removed

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

def load_counter() -> int:
    if LIMIT_FILE.exists():
        return json.loads(LIMIT_FILE.read_text()).get("count", 0)
    return 0

def save_counter(n: int):
    LIMIT_FILE.write_text(json.dumps({"count": n}))

def increment_runs() -> bool:
    n = load_counter()
    if n >= RUN_LIMIT:
        return False
    save_counter(n + 1)
    return True

def scrape_page(url: str, wait_ms: int = 2000) -> str:
    """Scrapes a single URL using Firecrawl, returning Markdown content."""
    # Potential future improvements:
    # - Explore `pageOptions={'onlyMainContent': True}` if switching to JSON format.
    # - Use Firecrawl 'actions' for pages requiring complex interaction.
    # - More specific error handling for Firecrawl exceptions.
    try:
        res = firecrawl.scrape_url(url, formats=["markdown"], waitFor=wait_ms)
        md = res.markdown if res else ""
        if not md:
            # More specific error message
            raise RuntimeError(f"🔥 No Markdown text extracted from {url}. The page might be empty, block scraping, or require longer 'comprehensive' wait time.")
        return md
    except Exception as e:
        # Catch potential Firecrawl specific errors if the SDK defines them, otherwise generic
        # Log the error for server-side debugging if needed
        # print(f"Scraping Error for {url}: {e}")
        raise RuntimeError(f"Scraping failed for {url}. Reason: {e}")

# Refined Agent Prompt V4 (Person-Centric Sales Focus)
AGENT_PROMPT = textwrap.dedent("""\
    You are an expert AI copywriter specializing in crafting compelling, personalized cold email opening lines. Your primary function is to analyze provided web page text and generate ONE unique opening sentence. Accuracy, relevance, **natural flow**, and adherence to source text are paramount.

    INPUT FORMAT:
    - PAGE_URL: The URL where the text originated.
    - PAGE_TEXT: The content scraped from the page.
    - EMAIL_PURPOSE: (Optional) The reason for sending the email, which might include selling a product/service.

    CRITICAL CONSTRAINTS:
    - **STRICTLY ADHERE TO PAGE_TEXT:** Your generated sentence MUST be based *exclusively* on information explicitly present in the provided PAGE_TEXT.
    - **NO HALLUCINATION / INVENTION:** Under NO circumstances should you invent facts, infer details not present, speculate, or use any external knowledge beyond interpreting the provided text. Accuracy to the source is vital.
    - **FALLBACK MECHANISM:** If, after careful analysis, you cannot identify a *specific*, *compelling*, and *relevant* point within the PAGE_TEXT to base a personalized opening line upon (considering the EMAIL_PURPOSE if provided), your *only* output MUST be the exact phrase: `No specific opening line found based on the provided text.` Do not output this phrase if you *can* find a suitable point.

    YOUR TASK:
    1.  **Identify Subject & Purpose:** Determine if the `PAGE_TEXT` primarily describes an individual person or an organization. Understand the `EMAIL_PURPOSE`, especially if it involves sales or partnership.
    2.  **Scrutinize PAGE_TEXT:** Carefully search the text for 1-2 *specific* and *verifiable* details (e.g., a recent company announcement title, a quantifiable achievement mentioned, a unique phrase from their mission/values, a specific project/product name, a directly stated company goal or challenge, specific role/expertise/accomplishment). Avoid generic marketing language.
    3.  **Assess Relevance & Connection:** Evaluate the identified detail(s):
        *   Is it specific and non-generic?
        *   **If EMAIL_PURPOSE is provided (especially sales/partnership):** Does the detail *directly relate* to the product, service, or collaboration mentioned in the purpose? Can you form a logical, *natural-sounding* bridge between the person's/company's work (from the text) and the value proposition (from the purpose)?
    4.  **Generate OR Fallback:**
        *   **If** a specific, relevant detail connecting to the purpose (or compelling on its own) is found: Craft a *single*, concise sentence (strict maximum 25 words). **This sentence MUST sound natural, flow smoothly, and be engaging – avoid robotic phrasing.** If selling to a person, elegantly connect the detail about *them* or *their work* to the potential benefit or relevance of your offering for *them*. Ensure it works perfectly as a cold email's *very first sentence*.
        *   **Else (No suitable detail/connection found):** Output the exact fallback phrase: `No specific opening line found based on the provided text.`
    5.  **Output ONLY the Result:** Your entire response must be *either* the single generated sentence *or* the exact fallback phrase. No explanations, labels, quotes, or other text.

    Example 1 (Selling to Person - Improved Flow):
    PAGE_URL: https://www.dbfwclegal.com/jim-y-wong/
    PAGE_TEXT: "...His business law practice encompasses...general regulatory and compliance (including visa applications and immigration compliance)..."
    EMAIL_PURPOSE: Introduce our AI platform that helps streamline immigration case workflows and boost client engagement only for O1 and EB1A visa types.

    Example Output 1:
    Your background including immigration compliance caught my eye; we're helping attorneys streamline O1/EB1A workflows with AI and thought it might interest you.
    *(Alternative Output 1):*
    Saw your profile mentions specific experience with immigration compliance – our AI platform simplifies O1/EB1A workflows, which seemed highly relevant.

    Example 2 (General Outreach to Company):
    PAGE_URL: https://example.com/about
    PAGE_TEXT: "...Our mission is to revolutionize widget production using sustainable AI..."
    EMAIL_PURPOSE: (None provided)

    Example Output 2:
    I was impressed to read about ExampleCorp's mission to revolutionize widget production using sustainable AI.

    Example 3 (Fallback):
    PAGE_URL: https://example.com/contact
    PAGE_TEXT: "Contact us via phone or email. Our address is 123 Main St."
    EMAIL_PURPOSE: Sales outreach

    Example Output 3:
    No specific opening line found based on the provided text.
""")

# Define the Personalization Agent globally
personalization_agent = Agent(
    name="Personalization Agent",
    instructions=AGENT_PROMPT,
    model="gpt-4.1" # Or preferred model like gpt-4o
)

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
    st.header("Analyze Page Content") # Updated header

    # ----- URL + options ----- 
    url = st.text_input("Enter profile/company URL...", placeholder="https://example.com/about") # Updated placeholder
    # --- Add Email Purpose Input --- 
    email_purpose = st.text_input("Purpose of email (optional context):", placeholder="e.g., Job application, Sales pitch")
    # --- End Email Purpose Input --- 
    use_wait = st.checkbox("Use comprehensive text extraction (wait for JS)")
    # Renamed button text
    create = st.button("✨ Generate Opening Line", use_container_width=True)
    
    st.divider()

    # ----- Email collection (optional) ----- 
    # (Kept as is, still potentially useful)
    st.subheader("Join the wait‑list") 
    st.markdown("""
        Get notified about new features like:
        *   Multi-Page Analysis
        *   Purpose-Driven Lines
        *   CRM Integration
    """)
    with st.form("email_form"):
        email = st.text_input("Email address", placeholder="you@example.com")
        agree = st.checkbox("I agree to receive product updates")
        if st.form_submit_button("Subscribe"):
            if "@" in email and "." in email and agree:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    with open(DATA_DIR / "emails.csv", "a", encoding="utf‑8") as f:
                        f.write(f"{email}\n")
                    st.success("✅ Thanks for subscribing!")
                except Exception as e:
                    st.error(f"Could not save email: {e}")
            elif not agree:
                st.error("Please agree to receive updates to subscribe.")
            else:
                st.error("Please enter a valid email address.")

    st.divider()
    # ----- Feedback Form (Moved to Sidebar) -----
    with st.expander("Provide Feedback (Optional)"):
        feedback_text = st.text_area("Share your experience or suggestions:", key="feedback_text_area")
        if st.button("Submit Feedback", key="feedback_submit_button"):
            if feedback_text:
                try:
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(FEEDBACK_FILE, "a", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                        writer.writerow([timestamp, feedback_text]) 
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    st.error(f"Could not save feedback: {e}")
            else:
                st.warning("Please enter some feedback before submitting.")

# ---- Main pane ----
st.title(f"✨ {APP_NAME}") # Updated icon
st.markdown('<p class="subheader">Generate personalized cold email opening lines from web content.</p>', unsafe_allow_html=True) # Updated subheader

# Generation Logic (triggered by button click in sidebar)
if create:
    # Reset previous generation result
    st.session_state.generated_line = None 
    # Clear previous page text state if create is clicked again
    st.session_state.page_text = None 
    st.session_state.source_url = None

    # Validate URL input
    if not url:
        st.warning("Please enter a URL in the sidebar.")
    elif len(url) > MAX_URL_LENGTH:
        st.error(f"URL is too long (max {MAX_URL_LENGTH} characters).")
    elif "://" not in url:
        st.error("Please enter a valid URL (including http:// or https://).")
    else:
        # --- Scrape Page --- 
        page_md = None
        try:
            scrape_wait_time = 4000 if use_wait else 2000 
            with st.spinner("Reading page... (please wait)" ):
                page_md = scrape_page(url, wait_ms=scrape_wait_time)
            st.session_state.page_text = page_md # Store scraped text
            st.session_state.source_url = url # Store URL
            # Reset session generation counter for the new page
            st.session_state.session_runs = 0 
            st.success("✅ Page content loaded.")
        except Exception as e:
            st.error(f"Could not read page: {e}")
        
        # --- Generate Opening Line (only if scraping succeeded) --- 
        if page_md:
            # Check limits before calling agent
            if st.session_state.session_runs < SESSION_GENERATION_LIMIT and increment_runs():
                with st.spinner("Generating opening line... (please wait)"):
                    try:
                        context_chunks = create_context_chunks(st.session_state.page_text, max_tokens=3000)
                        full_context = "\n\n---\n\n".join(context_chunks)[:80000] # Keep large context
                        
                        # --- Prepare Email Purpose String --- 
                        purpose_string = ""
                        if email_purpose: # Check if user entered anything
                            purpose_string = f"EMAIL_PURPOSE: {email_purpose}\n\n"
                        # --- End Email Purpose Prep --- 

                        # Prepare input_text including optional purpose
                        input_text = (
                            f"PAGE_URL: {st.session_state.source_url}\n\n"
                            f"{purpose_string}" # Add formatted purpose here
                            f"PAGE_TEXT:\n{full_context}"
                        )
                        
                        # Run the personalization agent
                        generated_line = asyncio.run(run_personalization_agent(personalization_agent, input_text))
                        st.session_state.generated_line = generated_line # Store result
                        
                        # Increment counter
                        st.session_state.session_runs += 1
                        
                        # Log usage (including purpose)
                        try:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # Add email_purpose to the log data
                            log_data = [ timestamp, st.session_state.source_url, email_purpose, generated_line ]
                            DATA_DIR.mkdir(parents=True, exist_ok=True)
                            with open(USAGE_LOG_FILE, "a", newline='', encoding="utf-8") as f:
                                writer = csv.writer(f)
                                # Optional: Update header if adding it
                                # if f.tell() == 0:
                                #    writer.writerow(["Timestamp", "SourceURL", "EmailPurpose", "GeneratedLine"])
                                writer.writerow(log_data)
                        except Exception as log_e:
                            st.warning(f"Could not log usage data: {log_e}")

                    except Exception as agent_e:
                        st.error(f"Generation error: {agent_e}")
                        st.session_state.generated_line = "Error generating line." # Set error message
            
            # Handle limits reached (display message, don't generate)
            elif st.session_state.session_runs >= SESSION_GENERATION_LIMIT:
                 st.warning(f"Session limit of {SESSION_GENERATION_LIMIT} generations reached for this URL analysis.")
            else: # Implies persistent limit reached
                 st.error("The overall service demo limit has been reached. Please try again later.")

# --- Display Output Area --- 
# This section runs regardless of the 'create' button state, showing the *last* result
if st.session_state.get("page_text") and st.session_state.get("source_url"):
    st.divider()
    st.info(f"Showing results for: {st.session_state.source_url}")
    # Optionally show session count
    st.caption(f"Generations this session for this URL: {st.session_state.session_runs}/{SESSION_GENERATION_LIMIT}")

    # Display the generated line if available
    if st.session_state.get("generated_line"):
        st.subheader("✨ Suggested Opening Line:")
        st.text_area("Copy this line:", value=st.session_state.generated_line, height=100, key="output_line")
    elif create: # If create was clicked but line is missing (e.g., due to limits or error before generation)
         if st.session_state.session_runs >= SESSION_GENERATION_LIMIT or load_counter() >= RUN_LIMIT:
              pass # Limit message already shown above
         else: 
              st.warning("Could not generate opening line. Check for errors above or try again.")
    
    # Add expander for scraped text here for context
    with st.expander("View Analyzed Page Content (Text Format)", expanded=False):
        st.text_area("Page Text:", value=st.session_state.page_text,
                    height=300, key="analyzed_text_view", disabled=True)

# Initial message if no analysis has been triggered yet
elif not create and not st.session_state.get("page_text"):
    st.info("👈 Enter a URL in the sidebar and click **Generate Opening Line** to start.")
