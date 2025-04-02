# streamlit_app.py (Refactored for Brevity)

import streamlit as st
import os
import requests
import random
import google.generativeai as genai
import google.ai.generativelanguage as glm
import datetime
from gtts import gTTS
import time
from typing import List, Optional, Tuple, Dict, Any
import traceback
import logging
import io
import smtplib
from email.message import EmailMessage
import socket
import subprocess
import json
from bs4 import BeautifulSoup

# --- Logging Setup ---
log_filename = 'app_log.log'
log_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Fn:%(funcName)s - %(message)s')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
logger.addHandler(log_handler)
logger.info(f"--- Streamlit App Start / Rerun ({datetime.datetime.now()}) ---")

# --- Configuration & Constants ---
st.set_page_config(page_title="💧 Water Saver Assistant (FC)", layout="wide", initial_sidebar_state="expanded", page_icon="💧")
MODEL_NAME = 'gemini-1.5-flash-latest'
CURRENT_YEAR = datetime.datetime.now().year
DEFAULT_WATER_TIPS_URL = "https://thewaterproject.org/water_conservation_tips"
WATER_TIPS = [] # In-memory cache
TIP_FILENAME = "water_conservation_tip.txt" # Server-side file
AUDIO_FILENAME_TEMP = "tip_generated_audio_temp.mp3"
LAST_SCRAPED_URL = ""

# --- State Initialization ---
def init_session_state():
    """Initializes required session state variables."""
    logger.debug("Initializing session state...")
    state_defaults = {
        'current_tip': "", 'current_audio_bytes': None, 'processing_message': "",
        'user_input_prompt': "", 'last_final_response': "", 'interaction_log': [],
        'gemini_model': None, 'gemini_chat': None, 'gemini_init_status': "pending",
        'gemini_main_status_message': None, 'session_saved_tips': [],
        'session_saved_audio': [], 'recipient_email': ""
    }
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    logger.debug("Session state initialization checks complete.")

# --- Gemini Model Initialization ---
def initialize_gemini():
    """Initializes the Gemini model and chat session using secrets."""
    logger.info("Attempting Gemini initialization...")
    if st.session_state.gemini_init_status == "success":
        logger.info("Gemini already initialized.")
        return

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("`GEMINI_API_KEY` not found in Streamlit secrets.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        st.session_state.gemini_model = model
        st.session_state.gemini_chat = model.start_chat(history=[]) # Start chat immediately
        st.session_state.gemini_init_status = "success"
        st.session_state.gemini_main_status_message = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        logger.info(f"Gemini Model ('{MODEL_NAME}') & chat session initialized successfully.")

    except Exception as e:
        detailed_error = traceback.format_exc()
        st.session_state.gemini_init_status = "error"
        st.session_state.gemini_main_status_message = f"🛑 **Error Initializing Gemini:** {str(e)}. Check API key/config. See logs."
        st.session_state.gemini_model = None
        st.session_state.gemini_chat = None
        logger.error(f"Gemini Initialization Error:\n{detailed_error}", exc_info=False)
        log_handler.flush()

# --- Utility Functions ---
def get_timestamp_dt() -> datetime.datetime:
    return datetime.datetime.now()

def get_timestamp_str(dt_obj: Optional[datetime.datetime] = None) -> str:
    if dt_obj is None: dt_obj = get_timestamp_dt()
    return dt_obj.strftime("%Y%m%d_%H%M%S_%f")

def log_interaction(message: str, level: str = "info"):
    """Logs message to file and adds timestamped entry to session state log list."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    display_message = message.replace('\n', ' ').strip()
    log_entry = f"[{timestamp}] {display_message}"
    if 'interaction_log' not in st.session_state: st.session_state.interaction_log = []
    st.session_state.interaction_log.append(log_entry)

    log_map = {"info": logger.info, "warning": logger.warning, "error": logger.error, "debug": logger.debug}
    log_func = log_map.get(level, logger.info)
    log_func(message)


# --- Gemini Function Calling Functions ---

def scrape_water_tips(url: str):
    """Scrapes water tips from URL, attempting various selectors."""
    global WATER_TIPS, LAST_SCRAPED_URL
    log_interaction(f"Scraping: {url}", level="info")
    try:
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'text/html'}
        response = requests.get(url, timeout=25, headers=headers, allow_redirects=True)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        tip_elements = []
        # Strategy 1: thewaterproject.org specific divs
        potential_containers_sqs = soup.find_all('div', class_='sqs-block-html')
        if potential_containers_sqs:
            for container in potential_containers_sqs:
                tip_elements.extend(container.find_all('li')) # Get all li within

        # Strategy 2: princeton.edu specific content area
        if not tip_elements:
            content_selectors = ['div.article-body', 'div.content', 'article', 'main']
            for selector in content_selectors:
                article_body = soup.select_one(selector)
                if article_body:
                    paragraphs = article_body.find_all('p')
                    tip_elements.extend([p for p in paragraphs if p.get_text(strip=True) and (p.get_text(strip=True)[0].isdigit() or p.get_text(strip=True)[0] in '*-•‣·' or len(p.get_text(strip=True).split()) >= 6)])
                    if tip_elements: break # Found potential tips

        # Fallback: all list items
        if not tip_elements:
            log_interaction("Using fallback: finding all 'li' tags.", level="warning")
            tip_elements = soup.find_all('li')

        if not tip_elements:
            log_interaction("No potential tip elements found.", level="warning")
            return False

        min_len = 20
        scraped_list_raw = [el.get_text(strip=True).replace('\xa0', ' ') for el in tip_elements if len(el.get_text(strip=True)) >= min_len]
        scraped_list = list(dict.fromkeys(scraped_list_raw)) # Remove duplicates

        if not scraped_list:
            log_interaction(f"No usable text found after filtering (min length {min_len}).", level="warning")
            return False

        WATER_TIPS = scraped_list
        LAST_SCRAPED_URL = url
        log_interaction(f"Scraped {len(WATER_TIPS)} tips from {url}.", level="info")
        return True

    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        log_interaction(f"Scraping Error: Request failed for {url} (Status: {status_code}): {e}", level="error")
        return False
    except Exception as e:
        log_interaction(f"Scraping Error: Unexpected error during parsing for {url}: {e}", level="error", exc_info=True)
        return False

def get_water_conservation_tip(url: str = None):
    """Retrieves a random tip, scraping if needed. Returns {'tip': str, 'error': bool}."""
    global WATER_TIPS, LAST_SCRAPED_URL, DEFAULT_WATER_TIPS_URL
    effective_url = url.strip() if url and isinstance(url, str) else DEFAULT_WATER_TIPS_URL
    log_interaction(f"get_water_conservation_tip(url='{effective_url}')", level="info")

    needs_scrape = not WATER_TIPS or (effective_url != LAST_SCRAPED_URL)
    if needs_scrape:
        log_interaction(f"Cache empty or URL differs. Triggering scrape.", level="info")
        if effective_url != LAST_SCRAPED_URL and WATER_TIPS: WATER_TIPS.clear() # Clear cache if URL changes
        scrape_success = scrape_water_tips(effective_url)
        if not scrape_success:
             msg = f"Sorry, error retrieving tips from source: {effective_url}."
             return {"tip": msg, "error": True}

    if WATER_TIPS:
        try:
            selected_tip = random.choice(WATER_TIPS)
            log_interaction(f"Returning random tip: '{selected_tip[:100]}...'", level="info")
            return {"tip": selected_tip, "error": False}
        except IndexError:
             msg = "Error selecting tip from cache."
             log_interaction(msg, level="error")
             return {"tip": msg, "error": True}
    else:
        msg = f"Sorry, no usable tips found at {effective_url}."
        log_interaction(msg, level="error")
        return {"tip": msg, "error": True}

def save_tip(tip: str):
    """Saves tip to file and session history. Returns {'status': str, 'error': bool}."""
    global TIP_FILENAME
    log_interaction(f"save_tip(tip='{tip[:60]}...')", level="info")
    min_len = 10
    if not tip or not isinstance(tip, str) or len(tip.strip()) < min_len:
        msg = f"Failed: Invalid/short tip (min {min_len} chars)."
        log_interaction(msg, level="warning")
        return {"status": msg, "filename": TIP_FILENAME, "error": True}

    cleaned_tip = tip.strip()
    overall_status = ""
    session_filename = None
    try:
        # Part 1: Save to file
        with open(TIP_FILENAME, "w", encoding='utf-8') as f:
            f.write(f"--- Saved Tip ---\nTimestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Tip Content:\n{cleaned_tip}\n--- End of Tip ---")
        file_msg = f"Tip saved to file '{TIP_FILENAME}'."
        log_interaction(file_msg, level="info")

        # Part 2: Add to session history
        try:
            ts_dt = get_timestamp_dt()
            session_filename = f"tip_{get_timestamp_str(ts_dt)}.txt"
            tip_entry = {"filename": session_filename, "content": cleaned_tip, "timestamp": ts_dt}
            if 'session_saved_tips' not in st.session_state: st.session_state.session_saved_tips = []
            st.session_state.session_saved_tips.insert(0, tip_entry)
            overall_status = f"{file_msg} Added to session history."
            log_interaction(f"Tip added to session history as '{session_filename}'.", level="info")
            return {"status": overall_status, "filename": TIP_FILENAME, "session_filename": session_filename, "error": False}
        except Exception as session_e:
            log_interaction(f"Error adding tip to session state: {session_e}", level="error")
            overall_status = f"{file_msg} FAILED to add to session history: {session_e}"
            return {"status": overall_status, "filename": TIP_FILENAME, "error": False} # File save succeeded

    except Exception as file_e:
        msg = f"Error saving tip to file '{TIP_FILENAME}': {file_e}"
        log_interaction(msg, level="error")
        return {"status": msg, "filename": TIP_FILENAME, "error": True}

def set_volume(level: int):
    """Attempts to set system volume via pactl (Linux). Returns {'status': str, 'error': bool}."""
    log_interaction(f"set_volume(level={level})", level="info")
    if not isinstance(level, int) or not 0 <= level <= 150:
         msg = "Error: Volume level must be integer 0-150."
         log_interaction(f"Invalid volume level: {level}", level="warning")
         return {"status": msg, "error": True}

    # Check if pactl exists
    try:
         subprocess.run(['which', 'pactl'], check=True, capture_output=True, timeout=2)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
         msg = "Warning: 'pactl' not found. Volume control likely unavailable."
         log_interaction(msg, level="warning")
         return {"status": msg, "error": False} # Environment limitation, not function error

    # Attempt to set volume
    try:
        sink_name = "@DEFAULT_SINK@" # Use default sink identifier
        command = ["pactl", "set-sink-volume", sink_name, f"{level}%"]
        log_interaction(f"Executing: {' '.join(command)}", level="debug")
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)

        if result.returncode == 0:
            msg = f"Attempted to set volume to {level}% via pactl."
            log_interaction(msg, level="info")
            return {"status": msg, "level_set": level, "error": False}
        else:
            error_details = result.stderr.strip() or result.stdout.strip() or f"Unknown pactl error (Code: {result.returncode})"
            msg = f"Error setting volume via pactl: {error_details}"
            log_interaction(msg, level="error")
            return {"status": msg, "error": True}

    except (subprocess.TimeoutExpired, Exception) as e:
        msg = f"Error during pactl execution: {e}"
        log_interaction(msg, level="error")
        return {"status": msg, "error": True}

def tell_tip():
    """Reads tip from file, generates TTS bytes, stores for playback & history. Returns status dict."""
    global TIP_FILENAME
    log_interaction("tell_tip()", level="info")
    tip_content = ""
    audio_bytes = None
    try:
        # Part 1: Read tip from file
        if not os.path.exists(TIP_FILENAME):
            msg = f"Error: Tip file '{TIP_FILENAME}' not found. Please 'save tip' first."
            log_interaction(msg, level="error")
            return {"status": msg, "error": True, "audio_bytes_generated": False}
        try:
            with open(TIP_FILENAME, "r", encoding='utf-8') as f: lines = f.readlines()
            tip_lines = [line.strip() for line in lines if line.strip() and not line.startswith('---') and "Timestamp:" not in line]
            tip_content = "\n".join(tip_lines).replace("Tip Content:", "").strip()
            if not tip_content: raise ValueError("No tip content found in file.")
            log_interaction(f"Read tip from '{TIP_FILENAME}': '{tip_content[:70]}...'", level="debug")
        except Exception as read_e:
             msg = f"Error reading/parsing tip file '{TIP_FILENAME}': {read_e}"
             log_interaction(msg, level="error")
             return {"status": msg, "error": True, "audio_bytes_generated": False}

        # Part 2: Generate audio bytes
        try:
            text_to_speak = f"Here is the saved water conservation tip: {tip_content}"
            audio_fp = io.BytesIO()
            tts = gTTS(text=text_to_speak, lang='en', slow=False)
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_bytes = audio_fp.read()
            if not audio_bytes: raise ValueError("gTTS resulted in empty audio bytes.")
            generation_msg = f"Audio generated successfully ({len(audio_bytes)} bytes)."
            log_interaction(generation_msg, level="info")
        except Exception as tts_e:
            msg = f"Error during gTTS audio generation: {tts_e}"
            log_interaction(msg, level="error")
            return {"status": msg, "error": True, "audio_bytes_generated": False}

        # Part 3: Store for immediate playback
        st.session_state.current_audio_bytes = audio_bytes
        log_interaction("Stored audio bytes for immediate playback.", level="debug")

        # Part 4: Add to session history
        try:
            ts_dt = get_timestamp_dt()
            session_filename = f"audio_{get_timestamp_str(ts_dt)}.mp3"
            audio_entry = {"filename": session_filename, "audio_bytes": audio_bytes, "timestamp": ts_dt}
            if 'session_saved_audio' not in st.session_state: st.session_state.session_saved_audio = []
            st.session_state.session_saved_audio.insert(0, audio_entry)
            overall_status = f"{generation_msg} Added to session history."
            log_interaction(f"Audio added to session history as '{session_filename}'.", level="info")
            return {"status": overall_status, "audio_bytes_generated": True, "session_filename": session_filename, "error": False}
        except Exception as session_e:
            log_interaction(f"Error adding audio to session state: {session_e}", level="error")
            overall_status = f"{generation_msg} FAILED to add to session history: {session_e}"
            return {"status": overall_status, "audio_bytes_generated": True, "error": False} # Audio generated OK

    except Exception as main_e:
        msg = f"Unexpected error in tell_tip: {main_e}"
        log_interaction(msg, level="error", exc_info=True)
        st.session_state.current_audio_bytes = None
        return {"status": msg, "error": True, "audio_bytes_generated": False}

# --- Define Tools for Gemini ---
logger.info("Defining Gemini tools...")
available_functions = {"get_water_conservation_tip": get_water_conservation_tip, "save_tip": save_tip, "set_volume": set_volume, "tell_tip": tell_tip}
tools_list = None
try:
    function_declarations = [
        glm.FunctionDeclaration( name="get_water_conservation_tip", description="Retrieves a random water conservation tip, optionally from a specific URL.", parameters=glm.Schema(type=glm.Type.OBJECT, properties={'url': glm.Schema(type=glm.Type.STRING, description=f"Optional URL. Defaults to {DEFAULT_WATER_TIPS_URL}.") })),
        glm.FunctionDeclaration( name="save_tip", description=f"Saves a given tip text to the file '{TIP_FILENAME}' and session history.", parameters=glm.Schema(type=glm.Type.OBJECT, properties={'tip': glm.Schema(type=glm.Type.STRING, description="The tip text to save.")}, required=['tip'])),
        glm.FunctionDeclaration( name="set_volume", description="Sets system audio volume (Linux/pactl only, 0-150%).", parameters=glm.Schema(type=glm.Type.OBJECT, properties={'level': glm.Schema(type=glm.Type.INTEGER, description="Volume percentage.")}, required=['level'])),
        glm.FunctionDeclaration( name="tell_tip", description=f"Reads the tip from '{TIP_FILENAME}' aloud using TTS and adds audio to history.")
    ]
    tools_list = [glm.Tool(function_declarations=function_declarations)]
    logger.info("Gemini tools defined successfully.")
except Exception as e:
     logger.error(f"CRITICAL: Error creating Gemini function declarations: {e}", exc_info=True)


# --- Run Initialization ---
init_session_state()
if st.session_state.gemini_init_status == "pending": initialize_gemini()

# --- Email Sending Function ---
def send_tip_email(recipient_email: str, email_content: str) -> Tuple[bool, str]:
    """Sends content via email using secrets."""
    log_interaction(f"Attempting email to {recipient_email}", level="info")
    try:
        sender_email = st.secrets["EMAIL_SENDER_ADDRESS"]
        sender_password = st.secrets["EMAIL_SENDER_PASSWORD"]
        smtp_server = st.secrets["SMTP_SERVER"]
        smtp_port = int(st.secrets["SMTP_PORT"])
    except (KeyError, ValueError, TypeError) as e:
        msg = f"🛑 Email Setup Error in secrets.toml: {e}"
        logger.error(msg); log_handler.flush(); return False, msg

    if not recipient_email or "@" not in recipient_email or "." not in recipient_email.split('@')[-1]:
        return False, "⚠️ Invalid recipient email address."
    if not email_content or not isinstance(email_content, str):
        return False, "⚠️ Cannot send empty email content."

    try:
        msg = EmailMessage()
        msg['Subject'] = "💧 Water Saver Assistant Info"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg.set_content(f"Hi there,\n\nInfo from the Water Saver Assistant:\n\n---\n{email_content.strip()}\n---\n\nBest,\nWater Saver Assistant")
    except Exception as e:
        msg = f"🛑 Error constructing email: {e}"; logger.error(msg, exc_info=True); return False, msg

    server = None
    try:
        logger.info(f"Connecting to SMTP: {smtp_server}:{smtp_port}")
        if smtp_port == 465: server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        else: server = smtplib.SMTP(smtp_server, smtp_port, timeout=30); server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        logger.info(f"Email sent successfully to {recipient_email}")
        return True, f"✅ Email sent successfully to {recipient_email}!"
    except (smtplib.SMTPAuthenticationError, smtplib.SMTPConnectError, smtplib.SMTPHeloError,
            smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused, smtplib.SMTPDataError,
            socket.gaierror, ConnectionRefusedError, TimeoutError, OSError, smtplib.SMTPException) as e:
        error_msg = f"🛑 Email Sending Failed: {type(e).__name__} - {e}"
        logger.error(error_msg, exc_info=True); log_handler.flush()
        return False, error_msg
    except Exception as e:
        error_msg = f"🛑 Email Sending Failed (Unexpected): {e}"
        logger.error(error_msg, exc_info=True); log_handler.flush()
        return False, error_msg
    finally:
        if server:
            try: server.quit()
            except Exception: pass

# --- Streamlit UI Layout ---

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3779/3779161.png", width=80)
    st.header("Assistant Status")
    st.markdown("---")
    st.subheader("🤖 Gemini Model")
    if st.session_state.gemini_init_status == "success":
        st.success(f"✅ Model (`{MODEL_NAME}`) loaded.")
        if not tools_list: st.error("⚠️ Tools FAILED to load.")
        else: st.caption("Function Calling Tools: Loaded")
    elif st.session_state.gemini_init_status == "error":
        st.error(st.session_state.get('gemini_main_status_message', "🛑 Initialization failed."))
    else: st.info("⏳ Initializing...")

    st.markdown("---")
    st.subheader("✉️ Email Setup")
    try:
        sender = st.secrets["EMAIL_SENDER_ADDRESS"]; _=st.secrets["EMAIL_SENDER_PASSWORD"]; _=st.secrets["SMTP_SERVER"]; _=int(st.secrets["SMTP_PORT"])
        st.success("✅ Secrets appear configured."); st.caption(f"Sender: {sender}")
    except Exception: st.warning("⚠️ Email secrets missing/invalid in secrets.toml.")

    st.markdown("---")
    st.info("Ask AI to 'set volume [0-100]' (Linux/pactl only).", icon="🔊")
    st.markdown("---")
    st.caption(f"Logs: `{log_filename}`")
    st.caption(f"FC Mode: {'✅' if tools_list else '❌'}")
    if os.path.exists("pages/2_📜_History.py"): st.page_link("pages/2_📜_History.py", label="View Session History", icon="📜")


# --- Main Page ---
st.title("💧 Water Conservation Assistant (FC)")
st.markdown("Ask for tips or actions like `save tip`, `tell tip`.")
if st.session_state.gemini_init_status == "error": st.error(st.session_state.gemini_main_status_message)
st.markdown("---")

col1, col2 = st.columns([2, 3])

with col1: # Input Column
    st.subheader("💬 Chat with the Assistant")
    user_prompt_input = st.text_area("Your request:", placeholder="e.g., Get a tip for showering.\nThen save the tip and read it aloud.", height=200, key="user_prompt_input_key", label_visibility="collapsed")
    send_disabled = (st.session_state.gemini_init_status != "success") or (not tools_list)
    send_tooltip = "Send request" if not send_disabled else "Cannot send: Model/Tools not ready."

    if st.button("➡️ Send Request", key="send_request_button", disabled=send_disabled, help=send_tooltip, use_container_width=True):
        if user_prompt_input.strip():
            st.session_state.current_audio_bytes = None
            st.session_state.last_final_response = ""
            st.session_state.interaction_log = []
            st.session_state.user_input_prompt = user_prompt_input.strip()
            st.session_state.processing_message = "⏳ Sending request..."
            log_interaction(f"User Prompt: '{st.session_state.user_input_prompt}'", level="info")
            if not st.session_state.gemini_chat:
                 logger.error("CRITICAL: Send clicked, but chat object missing.")
                 st.error("🛑 Error: Chat session unavailable. Reload page.")
                 st.session_state.processing_message = ""
                 st.stop()
            st.rerun()
        else:
            st.warning("⚠️ Please enter a request.")
    if send_disabled and st.session_state.gemini_init_status != "pending":
        st.warning("AI interaction disabled. Check init status.", icon="🤖")

# --- Processing Logic Block ---
if st.session_state.processing_message:
    message_placeholder = st.empty()
    message_placeholder.info(st.session_state.processing_message, icon="⏳")
    model = st.session_state.gemini_model
    chat = st.session_state.gemini_chat
    user_prompt = st.session_state.user_input_prompt

    if not model or not chat or not tools_list:
        st.error("🛑 Error: Model/Chat/Tools unavailable. Cannot process.")
        log_interaction("Aborted processing: Model/Chat/Tools not ready.", level="error")
        st.session_state.processing_message = ""
        st.stop()

    try:
        log_interaction(f">> Sending prompt: '{user_prompt[:80]}...'", level="info")
        st.session_state.processing_message = f"⏳ Asking Gemini..."
        message_placeholder.info(st.session_state.processing_message, icon="🤖")

        response = chat.send_message(user_prompt, tools=tools_list)
        log_interaction("<< Initial response received.", level="debug")

        # --- Function Call Loop ---
        max_turns = 10
        turn = 0
        while turn < max_turns:
            turn += 1
            function_call = None
            try:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    part = candidate.content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                    else: break # It's final text
                else: break # Empty response content
            except (AttributeError, IndexError, Exception) as e:
                 log_interaction(f"<< Error parsing response structure: {e}", level="warning")
                 break # Assume final text or error

            if function_call:
                fc_name = function_call.name
                fc_args = dict(function_call.args) if function_call.args else {}
                log_interaction(f"<< FC Request ({turn}): {fc_name}({json.dumps(fc_args)})", level="info")
                st.session_state.processing_message = f"⚙️ Running: {fc_name}(...)"
                message_placeholder.info(st.session_state.processing_message, icon="⚙️")

                if fc_name in available_functions:
                    try:
                        with st.spinner(f"Running action: {fc_name}..."):
                            func_response = available_functions[fc_name](**fc_args)
                        if not isinstance(func_response, dict): func_response = {} # Ensure dict
                        log_interaction(f">> Sending result of {fc_name}...", level="debug")
                        st.session_state.processing_message = f"✅ Ran {fc_name}. Sending result..."
                        message_placeholder.info(st.session_state.processing_message, icon="✅")
                        response_part = glm.Part(function_response=glm.FunctionResponse(name=fc_name, response=func_response))
                        response = chat.send_message(response_part, tools=tools_list)
                        log_interaction("<< Received response after sending result.", level="debug")
                    except Exception as e_exec:
                        error_detail = traceback.format_exc()
                        error_msg = f"Error executing function '{fc_name}': {str(e_exec)}"
                        log_interaction(f"ERROR executing {fc_name}: {error_detail}", level="error")
                        st.session_state.processing_message = f"⚠️ Error in {fc_name}. Notifying..."
                        message_placeholder.warning(st.session_state.processing_message, icon="⚠️")
                        error_part = glm.Part(function_response=glm.FunctionResponse(name=fc_name, response={"error": True, "status": error_msg}))
                        response = chat.send_message(error_part, tools=tools_list)
                        log_interaction(f">> Sent error for {fc_name}.", level="warning")
                        # Continue loop to let Gemini respond to the error
                else:
                    log_interaction(f"ERROR: Unknown function requested: {fc_name}", level="error")
                    st.error(f"🛑 Assistant requested unavailable function: '{fc_name}'.")
                    break # Stop if function doesn't exist

        if turn >= max_turns: log_interaction("Warning: Max interaction turns reached.", level="warning")

        # --- Process Final Response ---
        log_interaction("<< Processing final response.", level="info")
        final_text = ""
        try:
           if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
               final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
           if final_text:
               st.session_state.last_final_response = final_text
               log_interaction(f"Final Text: '{final_text[:100]}...'", level="info")
               # Heuristic for 'current_tip'
               if len(final_text.split()) > 5 and not final_text.lower().startswith(("ok", "sure", "done", "error")):
                    st.session_state.current_tip = final_text
           else:
                st.session_state.last_final_response = "[Assistant provided no final text.]"
                log_interaction("Warning: No text in final response.", level="warning")
                # Log finish reason etc. if needed for debugging empty responses
        except Exception as e_final:
             log_interaction(f"Error processing final response object: {e_final}", level="error")
             st.session_state.last_final_response = f"[Error processing final response: {e_final}]"

    except Exception as e_main:
        st.error(f"🛑 Critical interaction error: {e_main}")
        log_interaction(f"CRITICAL Interaction Error: {traceback.format_exc()}", level="error")
        st.session_state.last_final_response = f"Error: Interaction failed ({e_main})."
    finally:
        st.session_state.processing_message = ""
        message_placeholder.empty()
        log_interaction("--- End Interaction ---", level="info")
        log_handler.flush()
        st.rerun()


# --- Output Display Column (Right) ---
with col2:
    st.subheader("💡 Assistant Response & Actions")
    st.markdown("---")
    if not st.session_state.processing_message:
        with st.container(border=False):
            # Display Final Text
            final_response_text = st.session_state.get('last_final_response', '')
            if final_response_text:
                st.markdown("**Assistant:**")
                st.markdown(final_response_text)
            elif st.session_state.get('user_input_prompt'): st.warning("No final response received.")

            # Display Interaction Log
            interaction_log = st.session_state.get('interaction_log', [])
            if interaction_log:
                with st.expander("Interaction Details"):
                    st.code("\n".join(interaction_log), language=None)

            # Display Audio Player
            audio_bytes = st.session_state.get('current_audio_bytes', None)
            if audio_bytes:
                st.caption("🎧 Assistant generated audio:")
                st.audio(audio_bytes, format='audio/mp3')

            # Email Section
            st.markdown("---"); st.subheader("✉️ Send Last Response via Email")
            content_to_email = final_response_text or "[No content]"
            can_email = bool(final_response_text) and not content_to_email.startswith(("[Error", "[Assistant", "[No text"))
            try: _ = st.secrets["EMAIL_SENDER_ADDRESS"]; email_ok = True
            except Exception: email_ok = False; st.warning("Email secrets not configured.", icon="🔒")
            email_disabled = not can_email or not email_ok

            st.session_state.recipient_email = st.text_input("Recipient Email:", value=st.session_state.recipient_email, key="email_input", disabled=email_disabled, label_visibility="collapsed", placeholder="Recipient Email..." if not email_disabled else "Email disabled")
            email_help = "Send response via email." if not email_disabled else ("No valid content to send." if email_ok else "Email not configured.")
            if st.button("📧 Send Email Now", key="email_button", help=email_help, use_container_width=True, disabled=email_disabled):
                recipient = st.session_state.recipient_email.strip()
                if recipient:
                    with st.spinner("Sending email..."): success, msg = send_tip_email(recipient, content_to_email)
                    if success: st.success(msg)
                    else: st.error(msg)
                    log_handler.flush()
                else: st.warning("⚠️ Enter recipient email.")

    elif not st.session_state.get('user_input_prompt'):
         st.info("Enter a request on the left to start.", icon="👈")

# --- Footer ---
st.markdown("---")
st.caption(f"© {CURRENT_YEAR} Water Saver Assistant (FC) | Gemini | Abdullah F. Al-Shehabi")
log_handler.flush()
logger.info("--- Streamlit Run Complete ---")