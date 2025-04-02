
import streamlit as st
import os
import requests
import random
import google.generativeai as genai
import google.ai.generativelanguage as glm # Function Calling Schemas
import datetime
from gtts import gTTS
import time
from typing import Union, List, Optional, Tuple, Dict, Any
import traceback
import logging
import io # For handling audio bytes
import smtplib
from email.message import EmailMessage
import socket
import subprocess # For set_volume
import json       # For handling function call args/responses if needed
from bs4 import BeautifulSoup # For web scraping

# --- Logging Setup ---
# Configure logging to file and optionally console
log_filename = 'app_log.log' # Changed name slightly
log_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
# Enhanced formatter includes function name for better context
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Function: %(funcName)s - %(message)s')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__) # Use standard __name__
logger.setLevel(logging.INFO) # Set to INFO to capture function calls, debug for more detail
# Clear existing handlers during Streamlit reruns to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)
# Optional: Add console logging during development
# import sys
# stream_handler = logging.StreamHandler(sys.stdout)
# stream_handler.setFormatter(log_formatter)
# logger.addHandler(stream_handler)
logger.info(f"--- Streamlit App Start / Rerun ({datetime.datetime.now()}) ---")


# --- Configuration & Constants ---
st.set_page_config(
    page_title="💧 Water Saver Assistant (FC)", # Indicate Function Calling
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💧"
)

MODEL_NAME = 'gemini-1.5-flash-latest' # Use the desired Gemini model
CURRENT_YEAR = datetime.datetime.now().year

# --- Global Variables for Function Calling Functions ---
# Note: Global variables in Streamlit might reset between reruns or user sessions.
# State is best managed via st.session_state. File I/O persists on the server instance.
DEFAULT_WATER_TIPS_URL = "https://thewaterproject.org/water_conservation_tips"
WATER_TIPS = [] # In-memory cache for scraped tips (ephemeral)
TIP_FILENAME = "water_conservation_tip.txt" # File used by save_tip and tell_tip
AUDIO_FILENAME_TEMP = "tip_generated_audio_temp.mp3" # Temp file for gTTS if BytesIO fails
LAST_SCRAPED_URL = "" # Tracks the last URL scraped to avoid redundant requests

# --- State Initialization ---
def init_session_state():
    """Initializes all necessary session state variables if they don't exist."""
    logger.debug("Initializing session state...")
    # Core app state for UI and flow control
    if 'current_tip' not in st.session_state: st.session_state.current_tip = "" # Can store last relevant tip text
    if 'current_audio_bytes' not in st.session_state: st.session_state.current_audio_bytes = None # For immediate audio playback
    if 'processing_message' not in st.session_state: st.session_state.processing_message = "" # Displays current status (e.g., "Asking Gemini...")
    if 'user_input_prompt' not in st.session_state: st.session_state.user_input_prompt = "" # Stores the user's last submitted text
    if 'last_final_response' not in st.session_state: st.session_state.last_final_response = "" # Stores Gemini's final text response
    if 'interaction_log' not in st.session_state: st.session_state.interaction_log = [] # List to display interaction steps in UI

    # Gemini specific state
    if 'gemini_model' not in st.session_state: st.session_state.gemini_model = None # The generative model instance
    if 'gemini_chat' not in st.session_state: st.session_state.gemini_chat = None # The chat session object
    if 'gemini_init_status' not in st.session_state: st.session_state.gemini_init_status = "pending" # Tracks initialization: pending, success, error
    if 'gemini_main_status_message' not in st.session_state: st.session_state.gemini_main_status_message = None # For displaying init status/errors

    # History lists (for the separate History page)
    if 'session_saved_tips' not in st.session_state: st.session_state.session_saved_tips = [] # List of tip dictionaries [{filename, content, timestamp}]
    if 'session_saved_audio' not in st.session_state: st.session_state.session_saved_audio = [] # List of audio dictionaries [{filename, audio_bytes, timestamp}]

    # Email state
    if 'recipient_email' not in st.session_state: st.session_state.recipient_email = "" # Stores the recipient email address input

    logger.debug("Session state initialization checks complete.")

# --- Gemini Model Initialization Function ---
def initialize_gemini():
    """Initializes the Gemini model using API key from Streamlit secrets."""
    logger.info("Attempting Gemini initialization...")
    # Skip if already successfully initialized
    if st.session_state.gemini_init_status == "success":
        logger.info("Gemini already initialized successfully.")
        # Ensure a success message is set if missing
        if not st.session_state.get('gemini_main_status_message'):
             st.session_state.gemini_main_status_message = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        return

    api_key = None
    try:
        # Retrieve API key securely from Streamlit secrets
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            # Handle missing API key
            st.session_state.gemini_init_status = "error"
            error_msg = "🛑 **Error:** `GEMINI_API_KEY` not found in Streamlit secrets. AI features disabled. Please configure secrets."
            st.session_state.gemini_main_status_message = error_msg
            st.session_state.gemini_model = None
            st.session_state.gemini_chat = None
            logger.error("CRITICAL: GEMINI_API_KEY not found in Streamlit secrets.")
            log_handler.flush() # Ensure error is logged immediately
            return

        # Configure the Gemini library with the API key
        genai.configure(api_key=api_key)

        # Create the GenerativeModel instance
        # Optional: Add safety settings if needed (e.g., to be less restrictive, use with caution)
        # safety_settings = [ ... ]
        # model = genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings)
        model = genai.GenerativeModel(MODEL_NAME)
        logger.info(f"GenerativeModel instance created for '{MODEL_NAME}'.")

        # Store the model and update status
        st.session_state.gemini_model = model
        st.session_state.gemini_init_status = "success"
        success_msg = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        st.session_state.gemini_main_status_message = success_msg
        logger.info(f"Gemini Model ('{MODEL_NAME}') initialization successful.")

        # IMPORTANT: Initialize the chat session now that the model is ready
        # This allows reusing the chat history for follow-up messages if desired
        st.session_state.gemini_chat = model.start_chat(history=[])
        logger.info("Gemini chat session started.")

    except Exception as e:
        # Handle any other errors during initialization
        detailed_error = traceback.format_exc()
        error_message = f"🛑 **Error Initializing Gemini:** {str(e)}. Please check API key and configuration. See logs for details."
        st.session_state.gemini_init_status = "error"
        st.session_state.gemini_main_status_message = error_message
        st.session_state.gemini_model = None
        st.session_state.gemini_chat = None
        logger.error(f"Gemini Initialization Error:\n{detailed_error}", exc_info=False) # Log full traceback
        log_handler.flush()

# --- Utility Functions ---
def get_timestamp_dt() -> datetime.datetime:
    """Gets the current timestamp as a timezone-naive datetime object."""
    return datetime.datetime.now()

def get_timestamp_str(dt_obj: Optional[datetime.datetime] = None) -> str:
    """Generates a sortable timestamp string (YYYYMMDD_HHMMSS_ffffff) from a datetime object or now."""
    if dt_obj is None: dt_obj = get_timestamp_dt()
    return dt_obj.strftime("%Y%m%d_%H%M%S_%f")

def log_interaction(message: str, level: str = "info"):
    """Adds a message to the interaction log list in session state and logs to file."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] # Include milliseconds
    # Sanitize message slightly for display (replace newlines)
    display_message = message.replace('\n', ' ').strip()
    log_entry = f"[{timestamp}] {display_message}"

    # Safely append to interaction log list in session state
    if 'interaction_log' not in st.session_state:
        st.session_state.interaction_log = [] # Initialize if missing
    try:
        st.session_state.interaction_log.append(log_entry)
    except Exception as e_log_append:
         logger.error(f"Failed to append to st.session_state.interaction_log: {e_log_append}") # Log error but don't crash

    # Log to the file logger based on the specified level
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    else: # Default case if level is unknown
        logger.info(f"({level}) {message}") # Log with level prefix


# --- Gemini Function Calling Functions ---
# These functions are called by the Gemini model via the interaction loop

def scrape_water_tips(url: str):
    """
    Scrapes water conservation tips from the given URL.
    Uses BeautifulSoup and attempts various selectors for robustness.
    Updates global WATER_TIPS cache upon success.
    Returns: True if successful, False otherwise.
    """
    global WATER_TIPS, LAST_SCRAPED_URL # Use global vars (be mindful in concurrent/multi-user apps)
    log_interaction(f"Attempting to scrape tips from: {url}", level="info")
    scraped_list = []
    try:
        # --- Send HTTP GET request with headers ---
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        response = requests.get(url, timeout=25, headers=headers, allow_redirects=True) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # --- Parse HTML content ---
        # Determine encoding; prefer apparent_encoding, fallback to utf-8
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser') # Use response.text which respects encoding
        log_interaction(f"Successfully fetched content (Status: {response.status_code}) from {url}", level="debug")

        # --- Selector Logic: Try multiple strategies to find tips ---
        tip_elements = []

        # Strategy 1: Specific to thewaterproject.org structure (divs containing lists)
        # Find divs likely containing the main content list.
        potential_containers_sqs = soup.find_all('div', class_='sqs-block-html')
        if potential_containers_sqs:
            log_interaction(f"Strategy 1: Found {len(potential_containers_sqs)} 'div.sqs-block-html' containers.", level="debug")
            for container in potential_containers_sqs:
                # Prefer direct list children (ul or ol)
                list_tag = container.find(['ul', 'ol'], recursive=False)
                if list_tag:
                    items = list_tag.find_all('li', recursive=False) # Direct li children only
                    if items:
                        log_interaction(f"Found {len(items)} direct list items in a list tag.", level="debug")
                        tip_elements.extend(items)
                else: # If no direct list, search deeper but risk getting nested items
                    items = container.find_all('li')
                    if items:
                         log_interaction(f"Found {len(items)} list items (potentially nested) within container.", level="debug")
                         tip_elements.extend(items)

        # Strategy 2: Specific to princeton.edu example (paragraphs in main article body)
        # Try this only if Strategy 1 yielded nothing
        if not tip_elements:
             log_interaction("Strategy 1 failed, trying Strategy 2 (content area paragraphs)", level="debug")
             # Common selectors for main content areas
             content_selectors = ['div.article-body', 'div.content', 'div.main-content', 'article', 'main']
             article_body = None
             for selector in content_selectors:
                 article_body = soup.select_one(selector) # Use CSS selector
                 if article_body:
                     log_interaction(f"Found content area using selector: '{selector}'", level="debug")
                     break # Stop searching once a potential area is found

             if article_body:
                  # Look for paragraphs ('p') that likely contain tips
                  paragraphs = article_body.find_all('p')
                  min_words_para = 6 # Minimum words for a paragraph to be considered a tip
                  potential_tips = [
                      p for p in paragraphs
                      if p.get_text(strip=True) and ( # Must have text
                          p.get_text(strip=True)[0].isdigit() or # Starts with digit
                          p.get_text(strip=True)[0] in '*-•‣·' or # Starts with bullet
                          len(p.get_text(strip=True).split()) >= min_words_para # Or reasonably long
                      )
                  ]
                  tip_elements.extend(potential_tips)
                  if potential_tips:
                      log_interaction(f"Found {len(potential_tips)} potential tip paragraphs in content area.", level="debug")

        # Fallback Strategy: Find *all* list items ('li') on the page if other strategies fail
        if not tip_elements:
            log_interaction("Strategies 1 & 2 failed, falling back to finding all 'li' tags.", level="warning")
            tip_elements = soup.find_all('li') # Get all list items anywhere on the page

        # --- Process and Filter Found Elements ---
        if not tip_elements:
            log_interaction(f"Warning: No potential tip elements (li, p) found using any method on {url}.", level="warning")
            return False # Indicate failure if nothing was found

        # Extract text, clean it, filter short items, and remove duplicates
        min_tip_length_chars = 20 # Increase minimum character length
        scraped_list_raw = []
        for tip_el in tip_elements:
             # Get text, replace non-breaking spaces, strip whitespace
             text = tip_el.get_text(strip=True).replace('\xa0', ' ').strip()
             if len(text) >= min_tip_length_chars:
                 scraped_list_raw.append(text)

        # Remove duplicates while preserving order (requires Python 3.7+)
        scraped_list = list(dict.fromkeys(scraped_list_raw))

        if not scraped_list:
            log_interaction(f"Warning: Found elements but they contained no usable text after filtering (min length {min_tip_length_chars}).", level="warning")
            return False # Indicate failure if filtering removes everything

        # --- Success: Update global cache and return ---
        WATER_TIPS = scraped_list # Update the in-memory cache
        LAST_SCRAPED_URL = url # Record the URL that was successfully scraped
        log_interaction(f"Successfully scraped and filtered {len(WATER_TIPS)} unique tips from {url}.", level="info")
        return True # Indicate success

    # --- Error Handling for Scraping ---
    except requests.exceptions.Timeout:
        log_interaction(f"Scraping Error: Timeout occurred while fetching URL {url}", level="error")
        return False
    except requests.exceptions.RequestException as e:
        # Log specific HTTP status code if available in the exception response
        status_code = e.response.status_code if e.response is not None else 'N/A'
        log_interaction(f"Scraping Error: Network/HTTP error for URL {url} (Status: {status_code}): {e}", level="error")
        return False
    except Exception as e: # Catch any other unexpected errors during scraping/parsing
        detailed_error = traceback.format_exc()
        log_interaction(f"Scraping Error: Unexpected error during scraping/parsing for {url}: {e}\n{detailed_error}", level="error")
        return False

def get_water_conservation_tip(url: str = None):
    """
    Retrieves a random water conservation tip, potentially scraping a URL if needed.
    Designed for Gemini Function Calling. Returns a dictionary with 'tip' and 'error'.
    """
    global WATER_TIPS, LAST_SCRAPED_URL, DEFAULT_WATER_TIPS_URL # Access global cache state
    # Determine the effective URL to use (provided or default)
    effective_url = url.strip() if url and isinstance(url, str) else DEFAULT_WATER_TIPS_URL
    log_interaction(f"[Function Call] get_water_conservation_tip(url='{effective_url}')", level="info")

    needs_scrape = False
    # Scrape if: 1) Cache is empty, or 2) A specific URL is requested that differs from the last cache.
    if not WATER_TIPS or (effective_url != LAST_SCRAPED_URL):
        log_interaction(f"Tip cache empty or URL differs. Triggering scrape for '{effective_url}'.", level="info")
        needs_scrape = True
        # If URL changed, clear the old cache before scraping
        if effective_url != LAST_SCRAPED_URL and WATER_TIPS:
            log_interaction(f"URL changed. Clearing previous tips cache from '{LAST_SCRAPED_URL}'.", level="debug")
            WATER_TIPS.clear()

    # --- Perform Scrape if Necessary ---
    if needs_scrape:
        log_interaction(f"---> Calling scrape_water_tips for: {effective_url}", level="debug")
        scrape_success = scrape_water_tips(effective_url) # This function logs its own details/errors
        if not scrape_success:
             # If scraping failed, return an informative error message to Gemini
             msg = f"Sorry, I encountered an error trying to retrieve tips from the source: {effective_url}. Please check the URL or try the default source."
             log_interaction(f"[Function Call Result] Scraping failed for {effective_url}.", level="error")
             # Return error=True to signal failure
             return {"tip": msg, "error": True}
        # If scrape was successful, WATER_TIPS and LAST_SCRAPED_URL are now updated

    # --- Select and Return a Tip from Cache ---
    if WATER_TIPS:
        try:
            selected_tip = random.choice(WATER_TIPS)
            log_interaction(f"[Function Call Result] Returning random tip: '{selected_tip[:100]}...'", level="info")
            # Return the selected tip with error=False
            return {"tip": selected_tip, "error": False}
        except IndexError: # Should not happen if WATER_TIPS is not empty, but safety check
             msg = "Sorry, an internal error occurred selecting a tip from the cache."
             log_interaction("[Function Call Result] Error: IndexError selecting from non-empty WATER_TIPS list.", level="error")
             return {"tip": msg, "error": True}
    else:
        # This case indicates scraping failed or yielded no results, even if scrape_success was True (e.g., filtering removed all)
        msg = f"Sorry, no usable water conservation tips could be found or extracted from the source: {effective_url}."
        log_interaction("[Function Call Result] No tips available in cache even after scrape attempt.", level="error")
        return {"tip": msg, "error": True}

def save_tip(tip: str):
    """
    Saves the provided tip text to a server-side file (TIP_FILENAME) AND
    adds an entry to the session state history list ('session_saved_tips').
    Designed for Gemini Function Calling. Returns a status dictionary.
    """
    global TIP_FILENAME # Use the global filename constant
    log_interaction(f"[Function Call] save_tip(tip='{tip[:60]}...')", level="info")

    # --- Validate Input Tip ---
    min_tip_len_save = 10 # Minimum characters for a tip to be considered valid for saving
    if not tip or not isinstance(tip, str) or len(tip.strip()) < min_tip_len_save:
        msg = f"Failed: No valid or sufficiently long tip (min {min_tip_len_save} chars) provided to save."
        log_interaction(f"[Function Call Result] Invalid tip provided for saving: '{tip[:60]}...'", level="warning")
        # Return error=True as the core action cannot proceed
        return {"status": msg, "filename": TIP_FILENAME, "error": True}

    cleaned_tip = tip.strip() # Use stripped version consistently
    file_save_msg = ""
    session_save_msg = ""
    overall_status = ""
    session_filename_created = None # Track filename added to session

    try:
        # --- Part 1: Save to Physical File (overwrites existing file) ---
        # This file is read by the 'tell_tip' function.
        timestamp_file = datetime.datetime.now().isoformat() # Record save time
        with open(TIP_FILENAME, "w", encoding='utf-8') as file:
            # Write structured content for clarity (optional)
            file.write(f"--- Saved Tip ---\n")
            file.write(f"Saved Timestamp: {timestamp_file}\n")
            file.write(f"Tip Content:\n{cleaned_tip}\n") # Write the cleaned tip
            file.write(f"--- End of Tip ---")
        file_save_msg = f"Tip successfully saved to server file: '{TIP_FILENAME}'."
        log_interaction(f"[Function Call Result] {file_save_msg}", level="info")

        # --- Part 2: Add Entry to Streamlit Session State History List ---
        # This populates the list shown on the History page.
        try:
            timestamp_dt_session = get_timestamp_dt() # Get current time
            timestamp_str_session = get_timestamp_str(timestamp_dt_session)
            # Create a unique filename *for this session entry*
            session_filename_created = f"tip_{timestamp_str_session}.txt"
            tip_entry = {
                "filename": session_filename_created,
                "content": cleaned_tip, # Store the cleaned tip
                "timestamp": timestamp_dt_session # Store the datetime object
            }

            # Ensure the session state list exists before appending/inserting
            if 'session_saved_tips' not in st.session_state:
                st.session_state.session_saved_tips = [] # Initialize if it was somehow lost

            # Insert at the beginning to show newest first on the History page
            st.session_state.session_saved_tips.insert(0, tip_entry)
            session_save_msg = f"Tip also added to session history as '{session_filename_created}'."
            log_interaction(f"[Function Call Result] {session_save_msg}", level="info")
            overall_status = f"{file_save_msg} Added to session history."
            # Both parts succeeded
            return {"status": overall_status, "filename": TIP_FILENAME, "session_filename": session_filename_created, "error": False}

        except Exception as session_e:
            # Handle error specifically during session state update
            session_save_msg = f"Error adding tip to session state: {session_e}"
            log_interaction(f"[Function Call Result] {session_save_msg}", level="error")
            overall_status = f"{file_save_msg} FAILED to add to session history: {session_e}"
            # Return error=False here: the primary action (file save for tell_tip) succeeded.
            # The History page might miss this entry, but the core functionality isn't broken.
            return {"status": overall_status, "filename": TIP_FILENAME, "error": False} # Note: error is False

    except Exception as file_e:
        # --- Handle error during physical file saving ---
        msg = f"Error saving tip to file '{TIP_FILENAME}': {file_e}"
        log_interaction(f"[Function Call Result] {msg}", level="error")
        # If file save fails, the function failed its primary purpose. Return error=True.
        # Do not attempt to update session state if file save failed.
        return {"status": msg, "filename": TIP_FILENAME, "error": True}

def set_volume(level: int):
    """
    Attempts to set system audio volume using the 'pactl' command (Linux specific).
    Designed for Gemini Function Calling. Returns a status dictionary.
    """
    log_interaction(f"[Function Call] set_volume(level={level})", level="info")

    # --- Validate Input Volume Level ---
    if not isinstance(level, int) or not 0 <= level <= 150: # pactl often allows > 100% (150% is common max)
         msg = "Error: Volume level must be an integer between 0 and 150."
         log_interaction(f"[Function Call Result] Invalid volume level provided: {level}", level="warning")
         return {"status": msg, "error": True} # Invalid input is an error

    # --- Check if 'pactl' Command is Available ---
    # Use subprocess.run with 'which' or 'command -v' to check if pactl exists in PATH
    check_command = ['which', 'pactl'] # 'which' is common, 'command -v pactl' is more POSIX standard
    try:
         pactl_path_check = subprocess.run(check_command, capture_output=True, text=True, check=True, timeout=2)
         pactl_path = pactl_path_check.stdout.strip()
         log_interaction(f"'pactl' command found at: {pactl_path}", level="debug")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
         # If 'which' fails or isn't found, pactl is likely unavailable
         msg = "Warning: 'pactl' command not found in system PATH. Volume control is likely unavailable on this system."
         log_interaction(msg, level="warning")
         # Return error=False: This isn't a failure of the function's logic, but an environment limitation.
         # Gemini doesn't need to retry; it should inform the user.
         return {"status": msg, "error": False}

    # --- Attempt to Set Volume using pactl ---
    try:
        # Optional: Get default sink name for better log messages
        try:
            sink_info = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=3, check=True)
            sink_name = sink_info.stdout.strip()
            log_interaction(f"Identified default audio sink: {sink_name}", level="debug")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            sink_name = "@DEFAULT_SINK@" # Use the fallback identifier if getting name fails
            log_interaction("Could not get default sink name, using '@DEFAULT_SINK@'.", level="debug")

        # Construct the command to set volume
        command = ["pactl", "set-sink-volume", sink_name, f"{level}%"]
        log_interaction(f"Executing command: {' '.join(command)}", level="debug")

        # Execute the command, capture output, don't check=True initially to handle errors
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)

        # Check the result's return code
        if result.returncode == 0:
            # Success
            msg = f"Attempted to set volume to {level}% for sink '{sink_name}' via pactl."
            log_interaction(f"[Function Call Result] {msg}", level="info")
            return {"status": msg, "level_set": level, "error": False}
        else:
            # Failure: pactl command returned an error
            error_message = result.stderr.strip() if result.stderr else result.stdout.strip() # Error might be on stdout
            if not error_message: error_message = f"Unknown pactl error (Return Code: {result.returncode})"
            msg = f"Error executing 'pactl set-sink-volume' (Code {result.returncode}) for sink '{sink_name}': {error_message}"
            log_interaction(f"[Function Call Result] {msg}", level="error")
            # Return error=True as the command execution failed
            return {"status": msg, "error": True}

    except subprocess.TimeoutExpired:
        msg = f"Error: Timeout expired while trying to execute 'pactl set-sink-volume'."
        log_interaction(f"[Function Call Result] {msg}", level="error")
        return {"status": msg, "error": True}
    except Exception as e: # Catch any other unexpected Python errors
        msg = f"An unexpected Python error occurred during volume setting: {e}"
        log_interaction(f"[Function Call Result] {msg}", level="error")
        return {"status": msg, "error": True}

def tell_tip():
    """
    Reads the tip from TIP_FILENAME, generates TTS audio bytes using gTTS,
    stores bytes in session state ('current_audio_bytes') for immediate playback,
    AND adds an entry to the session state audio history list ('session_saved_audio').
    Designed for Gemini Function Calling. Returns a status dictionary.
    """
    global TIP_FILENAME # Use the global filename constant for reading
    log_interaction("[Function Call] tell_tip()", level="info")
    tip_content = ""
    audio_bytes = None
    generation_msg = ""
    session_save_msg = ""
    overall_status = ""

    try:
        # --- Part 1: Read the Tip Content from the File ---
        if not os.path.exists(TIP_FILENAME):
            # If the file doesn't exist, the function cannot proceed.
            msg = f"Error: Tip file '{TIP_FILENAME}' not found. Cannot read tip. Please ask the assistant to 'save tip' first."
            log_interaction(f"[Function Call Result] {msg}", level="error")
            # Return error=True as the prerequisite file is missing.
            return {"status": msg, "error": True, "audio_bytes_generated": False}

        # Try reading and parsing the file
        try:
            with open(TIP_FILENAME, "r", encoding='utf-8') as file:
                lines = file.readlines()

            # Extract tip content, handling potential formatting from save_tip
            tip_lines = []
            in_tip_section = False
            # Look for markers written by save_tip
            for line in lines:
                stripped_line = line.strip()
                if stripped_line == "Tip Content:": # Start marker
                    in_tip_section = True
                    continue # Skip the marker line itself
                if stripped_line == "--- End of Tip ---": # End marker
                    in_tip_section = False
                    break # Stop reading
                if in_tip_section:
                    tip_lines.append(stripped_line) # Collect lines within markers

            # If markers weren't found or no content between them, try simpler approach:
            # Assume content is lines not starting with '---' or containing 'Timestamp:'
            if not tip_lines:
                 log_interaction("Could not find 'Tip Content:' marker, trying fallback file read.", level="debug")
                 tip_lines = [line.strip() for line in lines if not line.startswith('---') and "Timestamp:" not in line]

            tip_content = "\n".join(tip_lines).strip() # Join lines and strip whitespace

            if not tip_content:
                 # If file exists but content is empty or couldn't be extracted
                 msg = f"Error: Tip file '{TIP_FILENAME}' exists but appears empty or tip content could not be extracted."
                 log_interaction(f"[Function Call Result] {msg}", level="warning")
                 # Return error=True as there's nothing to speak
                 return {"status": msg, "error": True, "audio_bytes_generated": False}

            log_interaction(f"Successfully read tip content from {TIP_FILENAME}: '{tip_content[:70]}...'", level="debug")

        except Exception as read_e:
             # Handle errors during file reading/parsing
             msg = f"Error reading or parsing tip file '{TIP_FILENAME}': {read_e}"
             log_interaction(f"[Function Call Result] {msg}", level="error")
             return {"status": msg, "error": True, "audio_bytes_generated": False}

        # --- Part 2: Generate Audio Bytes using gTTS ---
        try:
            log_interaction(f"Generating audio for the tip using gTTS...", level="debug")
            # Prepare the text to be spoken, adding context
            text_to_speak = f"Here is the saved water conservation tip: {tip_content}"

            # Use BytesIO for efficient in-memory handling (preferred over temp file)
            audio_fp = io.BytesIO()
            tts = gTTS(text=text_to_speak, lang='en', slow=False)
            tts.write_to_fp(audio_fp) # Write audio data directly to the BytesIO buffer
            audio_fp.seek(0) # Reset buffer position to the beginning for reading
            audio_bytes = audio_fp.read() # Read all bytes from the buffer
            audio_fp.close() # Close the buffer

            if not audio_bytes:
                # Should not happen if write_to_fp succeeded, but check anyway
                 msg = "Error: gTTS processing completed but resulted in empty audio bytes."
                 log_interaction(f"[Function Call Result] {msg}", level="error")
                 return {"status": msg, "error": True, "audio_bytes_generated": False}

            generation_msg = f"Audio for tip generated successfully ({len(audio_bytes)} bytes)."
            log_interaction(f"[Function Call Result] {generation_msg}", level="info")

        except Exception as tts_e:
            # Handle errors during gTTS processing
            msg = f"Error during gTTS audio generation: {tts_e}"
            log_interaction(f"[Function Call Result] {msg}", level="error")
            return {"status": msg, "error": True, "audio_bytes_generated": False}


        # --- Part 3: Store Bytes in Session State for IMMEDIATE Playback ---
        # This makes the audio player on the main page work right after the interaction.
        st.session_state.current_audio_bytes = audio_bytes
        log_interaction("Stored generated audio bytes in 'current_audio_bytes' for immediate playback.", level="debug")

        # --- Part 4: Add Audio Entry to Session State HISTORY List ---
        # This populates the list shown on the History page.
        try:
            timestamp_dt_audio = get_timestamp_dt() # Get current time
            timestamp_str_audio = get_timestamp_str(timestamp_dt_audio)
            # Create a unique filename *for this session entry*
            session_audio_filename = f"audio_{timestamp_str_audio}.mp3"
            audio_entry = {
                "filename": session_audio_filename,
                "audio_bytes": audio_bytes, # Store the actual audio data bytes
                "timestamp": timestamp_dt_audio # Store the datetime object
            }

            # Ensure the session state list exists before appending/inserting
            if 'session_saved_audio' not in st.session_state:
                st.session_state.session_saved_audio = [] # Initialize if missing

            # Insert at the beginning for newest-first display on History page
            st.session_state.session_saved_audio.insert(0, audio_entry)
            session_save_msg = f"Audio also added to session history as '{session_audio_filename}'."
            log_interaction(f"[Function Call Result] {session_save_msg}", level="info")
            overall_status = f"{generation_msg} Added to session history."
            # Both generation and adding to history succeeded
            return {
                "status": overall_status,
                "audio_file_read": TIP_FILENAME, # Indicate which file was read
                "audio_bytes_generated": True, # Signal that bytes are ready
                "session_filename": session_audio_filename, # Filename used in history
                "error": False
            }

        except Exception as session_e:
            # Handle error specifically during session state update for history
            session_save_msg = f"Error adding audio entry to session state history: {session_e}"
            log_interaction(f"[Function Call Result] {session_save_msg}", level="error")
            overall_status = f"{generation_msg} FAILED to add to session history: {session_e}"
            # Return error=False here: audio was generated and is ready for playback (primary goal).
            # The History page might miss this entry, but the immediate function is okay.
            return {
                "status": overall_status,
                "audio_file_read": TIP_FILENAME,
                "audio_bytes_generated": True,
                "error": False # Note: error is False for the overall function call
            }

    except Exception as main_e:
        # Catch any other unexpected errors in the main try block
        detailed_error = traceback.format_exc()
        msg = f"An unexpected error occurred during the tell_tip process: {main_e}"
        log_interaction(f"[Function Call Result] {msg}\n{detailed_error}", level="error")
        st.session_state.current_audio_bytes = None # Clear potentially partial state
        return {"status": msg, "error": True, "audio_bytes_generated": False}


# --- Define Tools for Gemini ---
logger.info("Defining Gemini tools (available functions and declarations)...")
# Dictionary mapping function names (strings) to the actual Python function objects
available_functions = {
    "get_water_conservation_tip": get_water_conservation_tip,
    "save_tip": save_tip,
    "set_volume": set_volume,
    "tell_tip": tell_tip,
}

# List of FunctionDeclarations describing the tools to Gemini
tools_list = None # Initialize to None in case of errors
try:
    # Create FunctionDeclaration objects using Gemini schema types (glm)
    function_declarations = [
        glm.FunctionDeclaration(
            name="get_water_conservation_tip",
            description=(
                "Retrieves a random water conservation tip from a reliable online source. "
                "It can scrape the source URL if necessary. Use this when the user asks for a general "
                "or specific water saving tip (e.g., 'tip for showering', 'tip for gardening'). "
                "You can optionally specify a URL if the user provides one."
            ),
            parameters=glm.Schema(
                type=glm.Type.OBJECT,
                properties={
                    'url': glm.Schema(
                        type=glm.Type.STRING,
                        description=f"Optional URL to scrape for tips. If omitted, defaults to a standard source ({DEFAULT_WATER_TIPS_URL}). Provide if the user mentions a specific website (e.g., 'get tip from princeton.edu')."
                    )
                }
                # 'url' is optional, so no 'required' field needed here
            )
        ),
        glm.FunctionDeclaration(
            name="save_tip",
            description=(
                f"Saves a given text string (the water conservation tip) to a server-side file named '{TIP_FILENAME}'. "
                "This also adds the tip to the user's current session history list for viewing later. "
                "Use this function ONLY if the user explicitly asks to 'save the tip', 'remember this tip', or similar, "
                "referring to a tip that was just provided or discussed."
            ),
            parameters=glm.Schema(
                type=glm.Type.OBJECT,
                properties={
                    'tip': glm.Schema(
                        type=glm.Type.STRING,
                        description="The exact water conservation tip text to save. It should be the complete and relevant tip."
                    )
                },
                required=['tip'] # The 'tip' parameter is mandatory for this function to work
            )
        ),
         glm.FunctionDeclaration(
            name="set_volume",
            description=(
                "Attempts to set the system audio output volume level using the 'pactl' command. "
                "This generally only works on Linux systems where 'pactl' is installed and accessible. "
                "Requires an integer percentage. Inform the user this might not work on their system or if an error occurs."
            ),
            parameters=glm.Schema(
                type=glm.Type.OBJECT,
                properties={
                    'level': glm.Schema(
                        type=glm.Type.INTEGER,
                        description="Target volume percentage (e.g., 70). Typical range is 0-100, but pactl might accept up to 150."
                    )
                },
                required=['level'] # The 'level' parameter is mandatory
            )
        ),
         glm.FunctionDeclaration(
            name="tell_tip",
            description=(
                f"Reads the previously saved water conservation tip aloud using text-to-speech (TTS). "
                f"It retrieves the tip from the server file '{TIP_FILENAME}' (which must have been saved previously using the 'save_tip' function). "
                "It generates audio data and makes it available for playback in the application interface. Use this when the user asks to 'tell the tip', 'read the saved tip', or 'speak the tip'."
            )
            # No 'parameters' field needed as the function reads from a fixed file and takes no input from the model.
        ),
    ]
    # IMPORTANT: Wrap the list of declarations in a `glm.Tool` object for the API call
    tools_list = [glm.Tool(function_declarations=function_declarations)]
    logger.info("Successfully created Gemini function declarations and tools list.")

except NameError:
    # This error occurs if 'glm' (google.ai.generativelanguage) failed to import
    logger.error("CRITICAL: Failed to create Gemini function declarations - 'glm' is likely not defined (Import Error?). Function calling will be disabled.")
    tools_list = None # Ensure tools_list is None if creation failed
except Exception as e:
     # Catch any other unexpected errors during tool definition
     logger.error(f"CRITICAL: Error creating Gemini function declarations: {e}", exc_info=True)
     tools_list = None


# --- Run Initialization ---
# Initialize session state variables FIRST (e.g., set defaults)
init_session_state()
# Then, initialize Gemini (reads secrets, configures API, creates model/chat)
# This runs only if status is 'pending', preventing re-initialization on every rerun
if st.session_state.gemini_init_status == "pending":
    initialize_gemini()


# --- Email Sending Function ---
# (Kept mostly as provided, with slight logging/error handling enhancements)
def send_tip_email(recipient_email: str, email_content: str) -> Tuple[bool, str]:
    """Sends the provided content via email using credentials from secrets.toml."""
    log_interaction(f"Attempting to send email to {recipient_email}", level="info")

    # 1. Retrieve Credentials SECURELY
    try:
        sender_email = st.secrets["EMAIL_SENDER_ADDRESS"]
        # Use App Password if 2FA enabled on sender account
        sender_password = st.secrets["EMAIL_SENDER_PASSWORD"]
        smtp_server = st.secrets["SMTP_SERVER"]
        smtp_port = int(st.secrets["SMTP_PORT"]) # Ensure port is integer
        logger.debug("Email credentials retrieved from Streamlit secrets.")
    except KeyError as e:
        msg = f"🛑 Email Setup Error: Missing secret '{e}' in secrets.toml. Cannot send email."
        logger.error(msg); log_handler.flush()
        return False, msg
    except (ValueError, TypeError):
         msg = "🛑 Email Setup Error: SMTP_PORT in secrets.toml must be a valid number."
         logger.error(msg); log_handler.flush()
         return False, msg
    except Exception as e:
        msg = f"🛑 Email Setup Error: Could not read email configuration from secrets: {e}"
        logger.error(msg, exc_info=True); log_handler.flush()
        return False, msg

    # 2. Validate Recipient Email Address
    if not recipient_email or "@" not in recipient_email or "." not in recipient_email.split('@')[-1]:
        logger.warning(f"Invalid recipient email format provided: {recipient_email}")
        return False, "⚠️ Please enter a valid recipient email address."

    # 3. Validate Email Content
    if not email_content or not isinstance(email_content, str):
        logger.warning("Attempted to email empty or invalid content.")
        return False, "⚠️ Cannot send email with empty content."

    # 4. Construct the Email Message using email.message for better structure
    try:
        msg = EmailMessage()
        msg['Subject'] = "💧 Your Water Saving Info from the Assistant!"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        # Create a slightly more informative email body
        email_body = f"""
Hi there,

Here is the information you requested from the Water Saver Assistant:

---
{email_content.strip()}
---

Keep saving water!

Best regards,
Your Water Saver Assistant (via Function Calling)
(Session ID approx: {get_timestamp_str()})
"""
        msg.set_content(email_body)
        logger.debug("Email message constructed successfully.")
    except Exception as e:
        msg = f"🛑 Error constructing email message object: {e}"
        logger.error(msg, exc_info=True); log_handler.flush()
        return False, msg

    # 5. Send the Email via SMTP with robust error handling
    server = None # Initialize server variable to None
    try:
        logger.info(f"Attempting SMTP connection to: {smtp_server}:{smtp_port}")
        # Choose SMTP_SSL for port 465 (implicit TLS) or SMTP for port 587 (explicit TLS/STARTTLS)
        if smtp_port == 465:
             logger.debug("Using SMTP_SSL (Implicit TLS for port 465).")
             server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) # Use SMTP_SSL class
             # No need for server.starttls() with SMTP_SSL
        else:
             logger.debug(f"Using SMTP (Explicit TLS/STARTTLS expected on port {smtp_port}).")
             server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
             server.ehlo() # Greet server before starting TLS
             server.starttls() # Upgrade connection to secure TLS
             server.ehlo() # Re-greet server after TLS upgrade

        # Login and send the message
        logger.info(f"Attempting login as {sender_email}")
        server.login(sender_email, sender_password)
        logger.info(f"Login successful. Sending email message to {recipient_email}")
        server.send_message(msg) # Use send_message for EmailMessage objects
        logger.info(f"Email successfully sent to {recipient_email}")
        # server.quit() is called implicitly by 'with' block if used, but explicit quit is fine too
        # server.quit() # Ensure connection is closed
        return True, f"✅ Email sent successfully to {recipient_email}!"

    # --- Specific SMTP Error Handling ---
    except smtplib.SMTPAuthenticationError:
        error_msg = "🛑 Email Sending Failed: Authentication error. Please check sender email and password (or App Password if using 2FA) in secrets.toml."
        logger.error(error_msg); log_handler.flush()
        return False, error_msg
    except (socket.gaierror, smtplib.SMTPConnectError, ConnectionRefusedError, TimeoutError, OSError) as e:
        # Catch various network/connection related errors
        error_msg = f"🛑 Email Sending Failed: Could not connect to SMTP server '{smtp_server}:{smtp_port}'. Check server address, port, network connectivity, and firewall settings. Details: {e}"
        logger.error(error_msg); log_handler.flush()
        return False, error_msg
    except smtplib.SMTPHeloError as e:
        error_msg = f"🛑 Email Sending Failed: Server did not reply properly to HELO/EHLO greeting: {e}"
        logger.error(error_msg); log_handler.flush()
        return False, error_msg
    except smtplib.SMTPDataError as e:
         # Error during the transmission of the message body
         error_msg = f"🛑 Email Sending Failed: Server refused the message data (Code: {e.smtp_code}): {e.smtp_error}"
         logger.error(error_msg); log_handler.flush()
         return False, error_msg
    except smtplib.SMTPRecipientsRefused as e:
        # Server refused one or more recipient addresses
        refused_str = ", ".join([f"{rcpt}: {err.decode() if isinstance(err, bytes) else err}" for rcpt, err in e.recipients.items()])
        error_msg = f"🛑 Email Sending Failed: Server refused one or more recipients: {refused_str}"
        logger.error(error_msg); log_handler.flush()
        return False, error_msg
    except smtplib.SMTPSenderRefused as e:
        # Server refused the sender address
        sender_addr = e.sender.decode() if isinstance(e.sender, bytes) else e.sender
        error_msg = f"🛑 Email Sending Failed: Server refused the sender address '{sender_addr}' (Code: {e.smtp_code}): {e.smtp_error.decode() if isinstance(e.smtp_error, bytes) else e.smtp_error}"
        logger.error(error_msg); log_handler.flush()
        return False, error_msg
    except smtplib.SMTPException as e: # Catch other generic SMTP library errors
        error_msg = f"🛑 Email Sending Failed: An SMTP-related error occurred: {e}"
        logger.error(error_msg, exc_info=True); log_handler.flush()
        return False, error_msg
    except Exception as e: # Catch any other unexpected Python errors during the process
        error_msg = f"🛑 Email Sending Failed: An unexpected error occurred: {e}"
        logger.error(error_msg, exc_info=True); log_handler.flush()
        return False, error_msg
    finally:
        # Ensure the server connection is closed if it was successfully opened
        if server:
            try:
                server.quit()
                logger.debug("SMTP server connection closed.")
            except Exception as quit_e:
                logger.warning(f"Exception while closing SMTP server connection: {quit_e}") # Log but don't raise


# --- Streamlit UI Layout ---

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3779/3779161.png", width=80)
    st.header("Assistant Status")
    st.markdown("---")

    # Gemini Model Status Display
    st.subheader("🤖 Gemini Model")
    if st.session_state.gemini_init_status == "success":
        st.success(f"✅ **Status:** Model (`{MODEL_NAME}`) loaded.")
        try:
            st.caption(f"google-generativeai lib: `{genai.__version__}`")
        except Exception:
            st.caption("Could not get library version.")
        # Check if tools were also loaded successfully
        if not tools_list:
            st.error("⚠️ Function Calling Tools FAILED to initialize. Check logs.")
        else:
            st.caption("Function Calling Tools: Loaded")
    elif st.session_state.gemini_init_status == "error":
        # Display the specific error message from initialization
        error_msg_detail = st.session_state.get('gemini_main_status_message', "🛑 **Error:** Initialization failed.")
        st.error(error_msg_detail + f"\n\nCheck `{log_filename}` for technical details.")
    else: # 'pending'
        st.info("⏳ Initializing Gemini Model...")

    st.markdown("---")

    # Email Setup Check Display
    st.subheader("✉️ Email Setup")
    email_secrets_ok_sidebar = False # Flag
    try:
        # Check existence without storing sensitive info
        sender_addr_check = st.secrets["EMAIL_SENDER_ADDRESS"]
        _ = st.secrets["EMAIL_SENDER_PASSWORD"]
        _ = st.secrets["SMTP_SERVER"]
        _ = int(st.secrets["SMTP_PORT"])
        st.success("✅ Email secrets: Appear configured.")
        st.caption(f"Sender Address: {sender_addr_check}") # Display non-sensitive sender
        email_secrets_ok_sidebar = True
    except KeyError as e:
        st.warning(f"⚠️ Email secrets missing (`{e}`). Email sending disabled. Check `.streamlit/secrets.toml`.", icon="📄")
    except (ValueError, TypeError):
        st.warning("⚠️ `SMTP_PORT` in secrets must be a number. Email sending disabled.", icon="📄")
    except Exception as e:
         st.error(f"🛑 Error checking email secrets: {e}. Email sending potentially disabled.", icon="📄")

    st.markdown("---")

    # Volume Control Info
    st.subheader("🔊 Volume Control")
    st.info("Ask the AI to set volume (e.g., 'set volume to 60'). Note: This uses 'pactl' and usually only works if the app is run on a Linux system with PulseAudio.", icon="🐧")
    st.markdown("---")

    # App Info & Links
    st.caption(f"Logs are written to `{log_filename}`")
    st.caption(f"Function Calling Mode: {'✅ Enabled' if tools_list else '❌ Disabled'}")
    # Link to the History Page (assumes it's in a 'pages' subdirectory)
    try:
        # Check if the file exists before creating the link (optional robustness)
        if os.path.exists("pages/2_📜_History.py"):
             st.page_link("pages/2_📜_History.py", label="View Session History", icon="📜")
        else:
             st.caption("History page file not found.")
    except Exception as e_pagelink:
         logger.warning(f"Could not create page link for History: {e_pagelink}")
         st.caption("Could not link to History page.")


# --- Main Page Content ---
st.title("💧 Water Conservation Assistant (Function Calling)")
st.markdown(
    "Interact with the AI assistant. You can ask for water-saving tips or request actions:\n"
    "*   `Get a tip about [topic]` (e.g., gardening, showering)\n"
    "*   `Get a tip from [URL]`\n"
    "*   `Save the last tip`\n"
    "*   `Tell me the saved tip` (reads it aloud)\n"
    "*   `Set volume to [percentage]` (e.g., 75)"
)

# Display Main Gemini Initialization Status/Errors prominently if needed
if 'gemini_main_status_message' in st.session_state and st.session_state.gemini_main_status_message:
    status_msg = st.session_state.gemini_main_status_message
    if status_msg:
        # Show success only if tools also loaded, otherwise show warning/error
        if st.session_state.gemini_init_status == "success":
            if tools_list:
                pass # Success message already in sidebar, avoid duplication
                # st.success(status_msg) # Can uncomment if redundancy is desired
            else:
                # Model loaded but tools failed
                st.error("⚠️ Model loaded, but Function Calling Tools FAILED to initialize. Functionality will be limited. Check logs.")
        elif st.session_state.gemini_init_status == "error":
            st.error(status_msg) # Show critical errors clearly
        # Option to clear message after first display:
        # st.session_state.gemini_main_status_message = None


st.markdown("---") # Separator before main layout

# --- Define Main Page Layout (Input and Output Columns) ---
col1, col2 = st.columns([2, 3]) # Input column (col1) slightly smaller than output (col2)

# --- Input Column (Left) ---
with col1:
    st.subheader("💬 Chat with the Assistant")
    st.markdown("Enter your request below:")

    # Text area for user input
    user_prompt_input = st.text_area(
        "Your request:",
        placeholder=(
            "Examples:\n"
            "- Get me a water saving tip.\n"
            "- Get a tip about washing clothes efficiently.\n"
            "- Get tip from https://environment.princeton.edu/news/10-simple-ways-conserve-water\n"
            "- Now save that tip.\n"
            "- Tell me the tip you saved.\n"
            "- Set volume to 80 then tell tip."
        ),
        height=220, # Adjusted height for more examples
        key="user_prompt_input_key", # Crucial for preserving input across minor reruns
        label_visibility="collapsed" # Hide default label, use subheader
    )

    # Determine button state: Disabled if Gemini or Tools failed init
    send_button_disabled = (st.session_state.gemini_init_status != "success") or (not tools_list)
    send_tooltip = "Send your request to the AI assistant."
    if send_button_disabled:
        send_tooltip = "Cannot send request: Gemini model or function calling tools failed to initialize. Check sidebar and logs."

    # The button that triggers the AI interaction
    if st.button("➡️ Send Request", key="send_request_button", disabled=send_button_disabled, help=send_tooltip, use_container_width=True):
        # Process only if user entered non-whitespace text
        if user_prompt_input.strip():
            # --- Prepare for new interaction ---
            # Reset relevant session state variables
            st.session_state.current_audio_bytes = None # Clear previous audio playback data
            st.session_state.last_final_response = ""  # Clear previous text response display
            st.session_state.interaction_log = []     # Start fresh interaction log display

            # Store the user's input prompt
            st.session_state.user_input_prompt = user_prompt_input.strip()

            # Set initial status message for the UI
            st.session_state.processing_message = "⏳ Sending request to Gemini..."

            # Log the start of this interaction cycle
            log_interaction(f"User Prompt Received: '{st.session_state.user_input_prompt}'", level="info")

            # --- Check if chat object is ready ---
            if not st.session_state.gemini_chat:
                 # This is a critical failure state if init didn't create the chat object
                 logger.error("CRITICAL: 'Send Request' clicked, but Gemini chat object ('gemini_chat') is missing in session state.")
                 st.error("🛑 Critical Error: Chat session not available. Initialization might have failed. Please reload the page or check logs.")
                 st.session_state.processing_message = "" # Clear processing msg on critical error
                 # Don't rerun, just stop
                 st.stop()

            # --- Trigger Streamlit rerun to execute the processing block ---
            logger.debug("Triggering rerun for processing block execution.")
            st.rerun()
        else:
            # User clicked button with empty input
            st.warning("⚠️ Please enter a request in the text area before sending.")

    # Optionally display a persistent warning if interaction is disabled
    if send_button_disabled and st.session_state.gemini_init_status != "pending":
        st.warning("AI interaction is disabled. Please check the initialization status in the sidebar.", icon="🤖")


# --- Processing Logic Block (Handles Gemini Interaction & Function Calling) ---
# This block executes only when 'processing_message' is set (i.e., after 'Send Request' triggers a rerun)
if st.session_state.processing_message:

    # Display the current status message using a placeholder for updates
    message_placeholder = st.empty()
    message_placeholder.info(st.session_state.processing_message, icon="⏳")

    # --- Get necessary components from session state ---
    model = st.session_state.gemini_model
    chat = st.session_state.gemini_chat
    user_prompt = st.session_state.user_input_prompt # The prompt for this cycle

    # --- Pre-flight checks ---
    if not model or not chat or not tools_list:
        error_msg = "🛑 Error: Cannot process request. Gemini model, chat session, or function tools are not available. Initialization may have failed."
        st.error(error_msg)
        log_interaction(error_msg, level="error")
        st.session_state.processing_message = "" # Clear message to stop loop
        # Don't rerun, allow user to see error
        st.stop()

    # --- Main Interaction Try-Except Block ---
    try:
        log_interaction(f">> Sending prompt to Gemini: '{user_prompt[:80]}...'", level="info")
        st.session_state.processing_message = f"⏳ Asking Gemini..." # Update status
        message_placeholder.info(st.session_state.processing_message, icon="🤖")

        # *** Initial send_message to Gemini with the user prompt and tools ***
        try:
            response = chat.send_message(user_prompt, tools=tools_list)
            # logger.debug(f"Initial response object: {response}") # Very verbose debugging
        except Exception as send_error_initial:
            # Handle errors during the API call itself (network, authentication, etc.)
            log_interaction(f"ERROR calling chat.send_message (initial): {send_error_initial}", level="error")
            st.error(f"🛑 Failed to communicate with Gemini: {send_error_initial}")
            # Attempt to get more details if available
            if hasattr(send_error_initial, 'response') and send_error_initial.response:
                 st.error(f"Details: {send_error_initial.response.text}")
            st.session_state.processing_message = "" # Clear status
            st.stop() # Stop processing if initial send fails

        log_interaction(f"<< Received initial response from Gemini.", level="debug")

        # --- Function Call Handling Loop ---
        max_turns = 10 # Safety limit for sequential function calls
        turn_count = 0
        while turn_count < max_turns:
            turn_count += 1
            log_interaction(f"--- Interaction Turn {turn_count}/{max_turns} ---", level="debug")
            function_call_to_execute = None # Reset flag for this turn

            # ** Step 1: Check response for a function call request **
            try:
                # Safely access the response structure
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                     # If Gemini returns empty content (e.g., safety blocked response before function call)
                     log_interaction("<< Response candidate has no content or parts. Exiting loop.", level="warning")
                     finish_reason_str = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else "Unknown"
                     if finish_reason_str != 'STOP':
                         st.warning(f"Assistant response was empty (Reason: {finish_reason_str}). Cannot proceed.")
                     break # Exit loop, will process as empty final response

                part = candidate.content.parts[0] # Assume call/text is in the first part

                # Check if this part IS a function call
                if hasattr(part, 'function_call') and part.function_call:
                    function_call_to_execute = part.function_call
                    # Parse arguments into a Python dictionary
                    args_dict = dict(function_call_to_execute.args) if function_call_to_execute.args else {}
                    api_request = {"name": function_call_to_execute.name, "args": args_dict}
                    log_interaction(f"<< Gemini requested Function Call: {api_request['name']}({json.dumps(api_request['args'])})", level="info")
                else:
                    # No function call in this part, means this is the final text response
                    log_interaction("<< No function call found in response part. Assuming final text response.", level="debug")
                    break # Exit the while loop to process the final text

            except (AttributeError, IndexError, ValueError, TypeError, Exception) as e_parse:
                 # Handle errors parsing the response structure from Gemini
                 log_interaction(f"<< Error parsing Gemini response structure or finding function call: {e_parse}", level="warning")
                 st.warning(f"⚠️ Could not properly parse Gemini's response structure. Attempting to process as text. Details: {e_parse}")
                 break # Exit loop and try to process whatever text might be there

            # --- Step 2: If a function call was detected, execute it ---
            if function_call_to_execute:
                function_name = api_request['name']
                args = api_request['args']

                # Update UI status message
                st.session_state.processing_message = f"⚙️ Assistant requested action: {function_name}(...)"
                message_placeholder.info(st.session_state.processing_message, icon="⚙️")

                # ** Look up the function in our available Python functions **
                if function_name in available_functions:
                    func_to_call = available_functions[function_name]

                    # ** Execute the local Python function **
                    function_response_data = None # Initialize response data
                    try:
                        log_interaction(f"   Executing local function: {function_name} with args: {args}", level="debug")
                        # Show a spinner in the UI during function execution
                        with st.spinner(f"Running action: {function_name}..."):
                            # Call the function, unpacking the arguments dictionary
                            function_response_data = func_to_call(**args)
                        log_interaction(f"   Local function '{function_name}' executed.", level="debug")
                        # logger.debug(f"   Raw return data from '{function_name}': {function_response_data}") # Can be verbose

                        # Validate: Ensure the function returned a dictionary for the API
                        if not isinstance(function_response_data, dict):
                             log_interaction(f"Warning: Function '{function_name}' did not return a dictionary! Returning empty dict to Gemini.", level="warning")
                             function_response_data = {} # Use empty dict as fallback

                        # ** Step 3: Send the function's result back to Gemini **
                        st.session_state.processing_message = f"✅ Action '{function_name}' completed. Sending result..."
                        message_placeholder.info(st.session_state.processing_message, icon="✅") # Update status
                        log_interaction(f">> Sending result of '{function_name}' back to Gemini...", level="debug")

                        # Construct the FunctionResponse Part using Gemini's schema
                        function_response_part = glm.Part(
                            function_response=glm.FunctionResponse(
                                name=function_name,
                                response=function_response_data # Send the dictionary returned by the function
                            )
                        )

                        # Send the result back and get Gemini's next response (might be another call or final text)
                        try:
                            response = chat.send_message(function_response_part, tools=tools_list)
                            log_interaction(f"<< Received response from Gemini after sending '{function_name}' result.", level="debug")
                            # logger.debug(f"Response object after sending result: {response}") # Verbose
                        except Exception as send_error_loop:
                             # Handle API errors when sending function result back
                             log_interaction(f"ERROR calling chat.send_message (in loop) after {function_name} result: {send_error_loop}", level="error")
                             st.error(f"🛑 Failed to send function result to Gemini: {send_error_loop}")
                             st.session_state.processing_message = "" # Clear status
                             st.stop() # Stop if we can't send result back

                    except Exception as exec_e:
                        # --- Handle errors during the execution of the local Python function ---
                        detailed_error = traceback.format_exc()
                        # Create a concise error message to send back to Gemini
                        error_msg_for_gemini = f"Error executing the '{function_name}' function in the application: {str(exec_e)}. The technical team has been notified via logs."
                        log_interaction(f"   ERROR executing local function '{function_name}': {exec_e}\n{detailed_error}", level="error")

                        # Update UI status to show error occurred
                        st.session_state.processing_message = f"⚠️ Error during action '{function_name}'. Notifying Assistant..."
                        message_placeholder.warning(st.session_state.processing_message, icon="⚠️")

                        # ** Send an error response back to Gemini **
                        # This tells Gemini the function failed, allowing it to potentially respond or retry differently.
                        log_interaction(f">> Sending ERROR response for '{function_name}' back to Gemini...", level="warning")
                        error_response_payload = { "error": True, "status": error_msg_for_gemini }
                        error_part = glm.Part(
                            function_response=glm.FunctionResponse(
                                name=function_name,
                                response=error_response_payload # Send structured error info
                            )
                        )

                        # Send the error back to Gemini
                        try:
                             response = chat.send_message(error_part, tools=tools_list)
                             log_interaction(f"<< Received response from Gemini after sending '{function_name}' ERROR result.", level="debug")
                        except Exception as send_error_loop_err:
                              # Handle API errors when sending the error notification itself
                              log_interaction(f"ERROR calling chat.send_message (in loop) after {function_name} execution error: {send_error_loop_err}", level="error")
                              st.error(f"🛑 Failed to notify Gemini about the function error: {send_error_loop_err}")
                              st.session_state.processing_message = "" # Clear status
                              st.stop() # Stop if we can't even report the error

                        # Decide whether to break loop on execution error. Let's continue,
                        # allowing Gemini to potentially comment on the error in its final response.
                        # break
                else:
                    # --- Handle case where Gemini requests a function not defined in our tools ---
                    error_msg = f"ERROR: Gemini requested an unknown function: '{function_name}'. This function is not defined in 'available_functions'."
                    log_interaction(error_msg, level="error")
                    st.error(f"🛑 Assistant tried to call an unavailable function ('{function_name}'). Please revise your request or check the tool definition.")
                    # Cannot proceed if the requested function doesn't exist.
                    st.session_state.processing_message = "" # Stop processing
                    break # Exit the loop immediately

        # --- End of Function Call Handling Loop ---

        # Check if loop exited due to reaching max turns
        if turn_count >= max_turns:
            log_interaction(f"Warning: Reached maximum interaction turns ({max_turns}). Exiting function call loop.", level="warning")
            st.warning(f"⚠️ Reached maximum interaction turns ({max_turns}). The conversation might be incomplete or stuck in a loop.")

        # ** Step 4: Process the Final Response from Gemini **
        # This should contain the final text response after all function calls are done.
        log_interaction("<< Processing final response from Gemini.", level="info")
        final_text = "" # Initialize final text variable
        try:
           # Safely access the text content from the last response object
           if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
               # Collect text from all parts (usually just one text part at the end)
               final_text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')]
               final_text = "".join(final_text_parts).strip() # Join parts and strip whitespace

           if final_text:
               # Store and log the final text
               log_interaction(f"Gemini Final Text Response: '{final_text[:200]}...'", level="info")
               st.session_state.last_final_response = final_text
               # Heuristic: Update 'current_tip' if the final response seems like a tip itself.
               # This helps the manual email function have relevant content.
               if len(final_text.split()) > 5 and not final_text.lower().startswith(("okay", "sure", "done", "alright", "i have", "error", "warning")):
                    st.session_state.current_tip = final_text
                    log_interaction("Stored final response text as 'current_tip' context.", level="debug")
               # else: # If it's just confirmation, don't overwrite current_tip context
               #      pass

           else: # Handle cases where the final response has no text content
                log_interaction("Warning: No text content found in the final response parts from Gemini.", level="warning")
                st.session_state.last_final_response = "[The assistant did not provide a final text response.]"
                # Check for reasons like safety blocking or abnormal finish
                finish_reason_str, safety_ratings_str, block_reason_str = "Unknown", "N/A", "None"
                try:
                    # Extract metadata if available
                    if response.candidates:
                         candidate = response.candidates[0]
                         if hasattr(candidate, 'finish_reason'): finish_reason_str = candidate.finish_reason.name
                         if hasattr(candidate, 'safety_ratings'):
                             ratings_summary = ", ".join([f"{r.category.name}: {r.probability.name}" for r in candidate.safety_ratings])
                             safety_ratings_str = f"[{ratings_summary}]"
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                         block_reason_str = response.prompt_feedback.block_reason.name

                    log_interaction(f"Final response metadata: Finish Reason={finish_reason_str}, Block Reason={block_reason_str}, Safety={safety_ratings_str}", level="debug")
                    # Display a warning in UI if finished abnormally or blocked
                    if finish_reason_str not in ['STOP', 'MAX_TOKENS'] or block_reason_str != 'NONE':
                         st.warning(f"⚠️ Assistant's final response was empty or potentially blocked (Finish: {finish_reason_str}, Block: {block_reason_str}).", icon="🤖")
                except Exception as e_final_meta:
                    log_interaction(f"Could not parse final response metadata: {e_final_meta}", level="warning")

        except Exception as e_final_proc:
             # Handle errors processing the final response object itself
             detailed_error = traceback.format_exc()
             log_interaction(f"ERROR processing final response object: {e_final_proc}\n{detailed_error}", level="error")
             st.session_state.last_final_response = f"[Error extracting final text from response: {e_final_proc}]"

    except Exception as e_main_try:
        # Catch any other unexpected errors during the main interaction try block
        detailed_error = traceback.format_exc()
        st.error(f"🛑 A critical error occurred during the chat interaction: {e_main_try}")
        log_interaction(f"CRITICAL: Unexpected error during main chat interaction block: {e_main_try}\n{detailed_error}", level="error")
        # Set an error message for the user
        st.session_state.last_final_response = f"Error: Interaction failed unexpectedly ({e_main_try}). Please check logs or try again."

    finally:
        # --- Cleanup after interaction attempt ---
        st.session_state.processing_message = "" # ALWAYS clear the processing status message
        message_placeholder.empty() # Remove the status message UI element
        log_interaction("--- End of Interaction Processing Block ---", level="info")
        log_handler.flush() # Ensure all logs for this interaction are written to file
        logger.debug("Triggering final rerun to update UI with results.")
        st.rerun() # Rerun Streamlit one last time to display the final state


# --- Output Display Column (Right) ---
with col2:
    st.subheader("💡 Assistant Response & Actions")
    st.markdown("---")

    # Display content only if the app is *not* actively processing a request
    if not st.session_state.processing_message:
        output_area = st.container(border=False) # Use container for grouping outputs

        with output_area:
            # 1. Display the Final Text Response from Gemini
            final_response_text = st.session_state.get('last_final_response', '')
            if final_response_text:
                st.markdown("**Assistant's Final Response:**")
                # Use st.markdown which supports simple formatting Gemini might use
                st.markdown(final_response_text)
            elif st.session_state.get('user_input_prompt'): # If interaction was attempted but no response
                 st.warning("No final text response was received from the assistant for the last request.")
            # else: # No interaction initiated yet
                 # st.info("Enter a request on the left and click 'Send Request' to start.", icon="👈")


            # 2. Display the Interaction Log (Expander)
            interaction_log = st.session_state.get('interaction_log', [])
            if interaction_log:
                st.markdown("---") # Separator
                with st.expander("Show Interaction Details (Function Calls, etc.)", expanded=False):
                    # Display log entries using st.code for readability
                    log_display_text = "\n".join(interaction_log)
                    st.code(log_display_text, language=None) # 'None' language prevents syntax highlighting

            # 3. Display Audio Player (if 'tell_tip' generated audio bytes)
            audio_bytes_to_play = st.session_state.get('current_audio_bytes', None)
            if audio_bytes_to_play:
                st.markdown("---") # Separator
                st.caption("🎧 Assistant generated the following audio:")
                try:
                    # Use st.audio to display the player using the bytes stored in session state
                    st.audio(audio_bytes_to_play, format='audio/mp3')
                    # Optional: Clear the audio bytes after displaying to prevent replay on next unrelated rerun
                    # Consider if user might want to replay it. For now, let it persist until next 'tell_tip'.
                    # st.session_state.current_audio_bytes = None
                except Exception as e_audio_player:
                    st.error(f"🛑 Error displaying audio player: {e_audio_player}")
                    logger.error(f"Audio Player Display Error", exc_info=True); log_handler.flush()

            # 4. Manual Email Section (Sends the last text response)
            st.markdown("---") # Separator
            st.subheader("✉️ Send Last Response via Email")

            # Content to email is the assistant's final text response
            content_to_email = final_response_text if final_response_text else "[No text content from assistant's last response]"
            # Basic validation for enabling the email button
            can_email_content = bool(final_response_text) and not content_to_email.startswith(("[Error", "Sorry", "[The assistant", "[No text"))

            # Check email secrets status again specifically for this UI section
            email_secrets_ok_main = False
            secrets_error_msg_main = ""
            try:
                st.secrets["EMAIL_SENDER_ADDRESS"] # Check required secrets
                #st.secrets["EMAIL_SENDER_PASSWORD"]
                #st.secrets["SMTP_SERVER"]
                int(st.secrets["SMTP_PORT"])
                email_secrets_ok_main = True
            except KeyError as e:
                 secrets_error_msg_main = f"Email disabled: Missing secret '{e}'."
            except (ValueError, TypeError):
                 secrets_error_msg_main = "Email disabled: SMTP_PORT must be a number."
            except Exception as e:
                 secrets_error_msg_main = f"Email disabled: Error checking secrets ({e})."

            # Determine overall disabled state for the email UI elements
            email_section_disabled = not can_email_content or not email_secrets_ok_main

            # Display warnings if email is disabled
            if not email_secrets_ok_main:
                 st.warning(f"{secrets_error_msg_main} Check `.streamlit/secrets.toml`.", icon="🔒")
            elif not can_email_content and st.session_state.get('user_input_prompt'): # Interaction happened but no valid content
                 st.warning("Cannot email: Assistant did not provide valid text content in the last response.", icon="📧")

            # Input field for Recipient's email address
            st.session_state.recipient_email = st.text_input(
                "Recipient Email:",
                placeholder="Enter recipient's email address" if not email_section_disabled else ("Generate a valid response first" if email_secrets_ok_main else "Email sending is not configured"),
                key="recipient_email_input_main", # Unique key
                value=st.session_state.recipient_email, # Persist input value
                disabled=email_section_disabled,
                label_visibility="collapsed" # Use placeholder
            )

            # Set appropriate help text for the Send Email button
            email_button_help = "Send the assistant's last text response via email."
            if not email_secrets_ok_main:
                email_button_help = "Email sending is disabled due to missing/invalid configuration in secrets.toml."
            elif not can_email_content:
                email_button_help = "Generate a valid text response from the assistant before sending via email."

            # The Send Email button
            if st.button("📧 Send Email Now", key="send_email_button_main",
                         help=email_button_help, use_container_width=True,
                         disabled=email_section_disabled):

                recipient = st.session_state.recipient_email.strip() # Get and clean recipient
                if recipient:
                     # Show spinner while sending
                     with st.spinner(f"Sending email to {recipient}..."):
                         success, msg = send_tip_email(recipient, content_to_email) # Call email function
                     # Display success or error message
                     if success:
                         st.success(msg)
                     else:
                         st.error(msg)
                     log_handler.flush() # Ensure email logs are flushed
                else:
                     # No recipient entered
                     st.warning("⚠️ Please enter a recipient email address first.")

    # Display a placeholder message if no interaction has been initiated yet in this session
    elif not st.session_state.get('user_input_prompt'):
         st.info("Enter a request on the left and click 'Send Request' to interact with the AI assistant.", icon="👈")


# --- Footer ---
st.markdown("---") # Final separator
st.caption(f"""
    © {CURRENT_YEAR} Water Saver Assistant (Function Calling Demo) | Powered by Google Gemini | Designed by **Abdullah F. Al-Shehabi**
""")

# Final log flush at the very end of the script execution/run
log_handler.flush()
logger.info("--- Streamlit App Run Complete ---")