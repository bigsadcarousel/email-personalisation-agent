import os, json, uuid, textwrap, asyncio, csv
from pathlib import Path
from typing import List
from datetime import datetime # Added for timestamping

import streamlit as st
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI
from agents import Agent, Runner
import tiktoken

# ----------------- ENV & CONSTANTS -----------------
load_dotenv()

APP_NAME    = "Uni-Agent (Beta V0)"
DATA_DIR    = Path(__file__).parent
LIMIT_FILE  = DATA_DIR / "limit.json"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"    # Path for feedback log
USAGE_LOG_FILE = DATA_DIR / "usage_log.csv"   # Path for usage log

OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
FIRECRAWL_KEY   = os.getenv("FIRECRAWL_API_KEY")
RUN_LIMIT       = int(os.getenv("RUN_LIMIT", "100")) # Persistent global limit

# New Limits
SESSION_RUN_LIMIT = 5     # Max questions per browser session
MAX_URL_LENGTH = 1000     # Prevent excessively long URLs
MAX_QUESTION_LENGTH = 500 # Prevent excessively long questions

if not (OPENAI_KEY and FIRECRAWL_KEY):
    st.stop("❌ Set OPENAI_API_KEY and FIRECRAWL_API_KEY in a .env file.")

firecrawl = FirecrawlApp(api_key=FIRECRAWL_KEY)
client    = OpenAI(api_key=OPENAI_KEY)

# Initialize session state for session run counter
if 'session_runs' not in st.session_state:
    st.session_state.session_runs = 0

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
        raise RuntimeError(f"Scraping failed for {url}: {e}")

# Refined Agent Prompt
AGENT_PROMPT = textwrap.dedent("""\
    You are a helpful and accurate Study Abroad Advisor AI. Your primary goal is to answer questions from prospective international students based *exclusively* on the provided web page text. Efficiency and clarity are key.

    INPUT FORMAT:
    The user input will contain:
    - PAGE_URL: The exact URL of the source page.
    - PAGE_TEXT: The text content scraped from that URL.
    - CONVERSATION HISTORY: (Optional) Recent messages exchanged in this session.
    - CURRENT QUESTION: The student's specific question.

    YOUR TASK:
    1. **Analyze the CURRENT QUESTION:** Identify the core intent and keywords.
    2. **Consider CONVERSATION HISTORY:** Use the history (if provided) to understand the context of the CURRENT QUESTION (e.g., pronouns, follow-ups).
    3. **Efficiently Search PAGE_TEXT:** Scan the provided text, focusing on sections relevant to the question's keywords and history context to find the answer quickly.
    4. **Extract Accurately:** Base your answer *only* on information explicitly stated in the PAGE_TEXT. Do *not* infer or use external knowledge.
    5. **Format Clearly:** Present the answer concisely. If using bullet points for lists, steps, or distinct pieces of information, include a maximum of 5 relevant points to maintain focus and readability.
    6. **Cite Source Once:** Begin your answer by stating the source URL using the format: `Based on the information provided [PAGE_URL]:` (replace PAGE_URL with the actual URL provided in the input). Do *not* repeat the citation within the answer.
    7. **Handle Missing Information:** If the answer is definitively *not* present in the PAGE_TEXT, respond ONLY with: `Based on the information provided [PAGE_URL]: I couldn't find specific information about [topic of the question].` (Replace bracketed text appropriately). Do not apologize excessively or offer to search elsewhere.
    8. **Maintain Tone:** Be professional, helpful, and direct. Avoid conversational filler.

    Example Answer Format (with Bullets - max 5 shown):
    Based on the information provided [https://example.edu/admissions/requirements]:
    * The application deadline for Fall 2025 is January 15th.
    * Required documents include official transcripts and two letters of recommendation.
    * There is an application fee of $75.
    * English proficiency test scores (TOEFL or IELTS) are required.
    * A personal statement outlining your goals is needed.

    Example Not Found Format:
    Based on the information provided [https://example.edu/financial-aid]: I couldn't find specific information about scholarship amounts for international students.
""")

# Define the Study Abroad Agent globally
study_abroad_agent = Agent(
    name="Study Abroad Assistant",
    instructions=AGENT_PROMPT,
    model="gpt-4.1"  # Using gpt-4o as per standard models
)

# Renamed and refined function to use Agents SDK
async def run_study_abroad_agent(agent: Agent, input_text: str) -> str:
    """Runs the study abroad agent with the provided input text."""
    try:
        result = await Runner.run(agent, input_text)
        return result.final_output
    except Exception as e:
        # Log error for debugging if needed
        # print(f"Agent run failed: {e}") 
        st.error(f"Error running agent: {e}")
        return "Sorry, I encountered an error trying to process your request."

# ----------------- STREAMLIT LAYOUT -----------------
st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")

# Custom theme colors
st.markdown("""
<style>
    .main-header {color: #8661c5;}
    .subheader {color: #6c7ac9;}
    .status-msg {background-color: #2d3047; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ("Create your agent") ----
with st.sidebar:
    st.header("Create your agent")

    # ----- URL + options (Moved Up) -----
    url = st.text_input("Enter university page URL", placeholder="https://example.edu/admissions")
    use_wait = st.checkbox("Use comprehensive text extraction (wait for JS)")
    create = st.button("Create agent", use_container_width=True)
    
    st.divider() # Add a visual separator

    # ----- Email collection (optional - Moved Down) -----
    st.subheader("Join the wait‑list")
    st.markdown("""
        Get notified about new features like:
        *   Multi-Page Intake
        *   Deadline & Scholarship Reminders
        *   End-to-End Workflow
    """)
    with st.form("email_form"):
        email = st.text_input("Email address", placeholder="you@example.com")
        agree = st.checkbox("I agree to receive product updates")
        if st.form_submit_button("Subscribe"):
            if "@" in email and "." in email and agree:
                # Ensure the data directory exists
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

# ---- Main pane ----
st.title(f"🤖 {APP_NAME}")
st.markdown('<p class="subheader">Your personal guide to university information worldwide.</p>', unsafe_allow_html=True)

# Display session run count (optional, for visibility)
# st.caption(f"Questions this session: {st.session_state.session_runs}/{SESSION_RUN_LIMIT}")

# Agent Creation Logic (triggered by button click)
if create:
    # Validate URL input before proceeding
    if not url:
        st.warning("Please enter a university page URL in the sidebar.")
    elif len(url) > MAX_URL_LENGTH:
        st.error(f"URL is too long (max {MAX_URL_LENGTH} characters).")
    elif "://" not in url:
        st.error("Please enter a valid URL (including http:// or https://).")
    else:
        # If URL is valid, proceed with scraping
        try:
            # Use longer wait time if comprehensive checkbox is ticked
            scrape_wait_time = 4000 if use_wait else 2000 
            with st.spinner(f"Reading university page (wait: {scrape_wait_time/1000}s)..." ):
                page_md = scrape_page(url, wait_ms=scrape_wait_time)
            st.session_state.page_text = page_md
            st.session_state.source_url = url
            st.session_state.session_runs = 0 
            st.session_state.messages = [] 
            st.success("✅ University content analyzed! Ask your questions below.")
        except Exception as e:
            st.error(f"Could not read page: {e}")
            if "page_text" in st.session_state: del st.session_state.page_text
            if "source_url" in st.session_state: del st.session_state.source_url
            if "messages" in st.session_state: del st.session_state.messages

# ------------- Chat Area ------------- 
# Only show chat if page text has been successfully loaded into session state
if "page_text" in st.session_state:
    
    # Add diagnostic expander - always visible when chat is active for a page
    with st.expander("View Page Content (Text Format)", expanded=False): # Keep it collapsed by default
        st.text_area("Page Text:", value=st.session_state.page_text,
                    height=300, key="scraped_text_view", disabled=True)

    # Initialize message list if it doesn't exist (should be handled by above logic, but belt-and-suspenders)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Calculate remaining questions for placeholder and disabled state
    questions_left = SESSION_RUN_LIMIT - st.session_state.session_runs
    placeholder_text = f"Ask a question... ({questions_left} left)" if questions_left > 0 else "Session limit reached for this page."
    chat_disabled = (questions_left <= 0)

    # Chat input
    prompt = st.chat_input(placeholder_text, disabled=chat_disabled, key=f"chat_input_{st.session_state.source_url}") 
    
    if prompt and not chat_disabled: 
        # Validate question length first
        if len(prompt) > MAX_QUESTION_LENGTH:
             with st.chat_message("assistant"):
                 st.error(f"Question is too long (max {MAX_QUESTION_LENGTH} characters).")
        else:
            # Display user message immediately
            st.session_state.messages.append({"role":"user", "content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- Check Limits --- 
            if st.session_state.session_runs < SESSION_RUN_LIMIT and increment_runs():
                # --- Limits OK - Proceed with Agent --- 
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing information..."):
                        try:
                            # Create context using the improved chunking function
                            context_chunks = create_context_chunks(st.session_state.page_text, max_tokens=3000)
                            # Significantly increase the character limit passed to the agent
                            full_context = "\n\n---\n\n".join(context_chunks)[:80000] # Increased limit
                            
                            # Prepare input for the agent, including history
                            history_string = ""
                            recent_messages = st.session_state.messages[-5:-1]
                            if recent_messages:
                                history_items = []
                                for msg in recent_messages:
                                    role = msg["role"].upper()
                                    content = msg["content"].replace('\n', ' ')
                                    history_items.append(f"{role}: {content}")
                                history_string = "CONVERSATION HISTORY:\n" + "\n".join(history_items) + "\n\n"

                            input_text = (
                                f"PAGE_URL: {st.session_state.source_url}\n\n"
                                f"PAGE_TEXT:\n{full_context}\n\n"
                                f"{history_string}"
                                f"CURRENT QUESTION: {prompt}"
                            )
                            
                            # Run agent
                            answer = asyncio.run(run_study_abroad_agent(study_abroad_agent, input_text))

                            # --- Log Usage Data --- 
                            try:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_data = [
                                    timestamp,
                                    st.session_state.source_url,
                                    prompt, # User's question
                                    answer # Agent's answer
                                ]
                                DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                                with open(USAGE_LOG_FILE, "a", newline='', encoding="utf-8") as f:
                                    writer = csv.writer(f)
                                    # Optional: Write header if file is new/empty
                                    # if f.tell() == 0:
                                    #     writer.writerow(["Timestamp", "SourceURL", "Question", "Answer"])
                                    writer.writerow(log_data)
                            except Exception as log_e:
                                st.warning(f"Could not log usage data: {log_e}") # Non-critical warning
                            # --- End Logging --- 

                            # Increment session counter *after* successful execution & logging
                            st.session_state.session_runs += 1
                            
                            # Update history and display answer
                            st.markdown(answer)
                            st.session_state.messages.append({"role":"assistant","content":answer})
                            
                            # Rerun script slightly to update chat input placeholder/state *immediately*
                            st.rerun()

                        except Exception as agent_e:
                            st.error(f"Analysis error: {agent_e}")
            # Handle limit reached cases cleanly after the check
            elif st.session_state.session_runs >= SESSION_RUN_LIMIT:
                 with st.chat_message("assistant"):
                     st.warning(f"Session limit of {SESSION_RUN_LIMIT} questions reached for this page analysis.")
                 st.stop()
            else: # Implies persistent limit reached
                 with st.chat_message("assistant"):
                     st.error("The overall service demo limit has been reached. Please try again later.")
                 st.stop()

    # ---- Feedback Form (Inside chat area block, but after main chat logic) ----
    st.divider()
    with st.expander("Provide Feedback (Optional)"):
        feedback_text = st.text_area("Share your experience or suggestions:", key="feedback_text_area")
        if st.button("Submit Feedback", key="feedback_submit_button"):
            if feedback_text:
                try:
                    DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Using CSV writer for proper escaping
                    with open(FEEDBACK_FILE, "a", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                        # Optional: Write header
                        # if f.tell() == 0:
                        #     writer.writerow(["Timestamp", "Feedback"])
                        writer.writerow([timestamp, feedback_text]) 
                    st.success("Thank you for your feedback!")
                    # Optionally clear the text area after submission
                    # st.session_state.feedback_text_area = "" 
                except Exception as e:
                    st.error(f"Could not save feedback: {e}")
            else:
                st.warning("Please enter some feedback before submitting.")

# Initial message if no page has been analyzed yet
elif not create: # Only show if the create button wasn't just clicked and page_text doesn't exist
    st.info("👈 Enter a university page URL in the sidebar and click **Create agent** to start exploring.")
