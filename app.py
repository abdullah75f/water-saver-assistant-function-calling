# app.py (Modified for multi-file saving)

import streamlit as st
import os
import requests # Keep for potential future web interaction if needed, though not used now
import random # Keep for potential future use
# import subprocess # Not used, removed
import google.generativeai as genai
# --- Function Calling Imports Removed ---
import datetime # Import datetime to get current year for copyright


from gtts import gTTS
# from bs4 import BeautifulSoup # Removed as scraper is gone
import time # For potential delays
# --- Add typing imports for Python 3.9 compatibility ---
from typing import Union, List, Optional, Tuple
import traceback # Import traceback for better error logging
import logging # Import standard logging
import sys # To help ensure flushing

# --- Setup Logging ---
# Try flushing aggressively
log_handler = logging.FileHandler('app_error.log', mode='a', encoding='utf-8')
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)


# --- Configuration & Constants ---
st.set_page_config(
    page_title="💧 Water Saver Assistant",
    layout="wide",
    initial_sidebar_state="expanded", # Keep sidebar open
    page_icon="💧"
)

# --- Constants ---
# REMOVE Single Filename Constants
# TIP_FILENAME = "water_conservation_tip_st.txt"
# AUDIO_FILENAME = "water_tip_audio_st.mp3"
MODEL_NAME = 'gemini-1.5-flash-latest' # Define model name centrally
CURRENT_YEAR = datetime.datetime.now().year # Get current year for copyright

# --- ADD Directory Constants and Path ---
APP_DIR = os.path.dirname(__file__) # Directory of app.py
SAVED_TIPS_DIR_NAME = "saved_tips"
SAVED_AUDIO_DIR_NAME = "saved_audio"
SAVED_TIPS_DIR = os.path.join(APP_DIR, SAVED_TIPS_DIR_NAME)
SAVED_AUDIO_DIR = os.path.join(APP_DIR, SAVED_AUDIO_DIR_NAME)


# --- State Initialization ---
# Use functions to avoid polluting global namespace and ensure init happens once
def init_session_state():
    if 'current_tip' not in st.session_state: st.session_state.current_tip = ""
    if 'audio_file_path' not in st.session_state: st.session_state.audio_file_path = None # Still useful for immediate playback
    if 'processing_message' not in st.session_state: st.session_state.processing_message = ""
    if 'direct_ai_prompt' not in st.session_state: st.session_state.direct_ai_prompt = ""
    if 'gemini_model' not in st.session_state: st.session_state.gemini_model = None
    if 'gemini_init_status' not in st.session_state: st.session_state.gemini_init_status = "pending" # pending, success, error
    # Separate message for main screen display
    if 'gemini_main_status_message' not in st.session_state: st.session_state.gemini_main_status_message = None

# --- Gemini Model Initialization Function ---
# Separated for clarity and to run only once if successful
def initialize_gemini():
    if st.session_state.gemini_init_status == "success":
        # Ensure main message is set correctly even if already initialized
        st.session_state.gemini_main_status_message = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        return # Already initialized successfully

    api_key = None
    try:
        # Primarily use Streamlit secrets
        api_key = st.secrets.get("GEMINI_API_KEY")

        # Fallback for local development (optional, comment out if not needed)
        # if not api_key:
        #     api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            st.session_state.gemini_init_status = "error"
            error_msg = "🛑 **Error:** `GEMINI_API_KEY` not found in Streamlit secrets or environment variables."
            st.session_state.gemini_main_status_message = error_msg # Set main message
            st.session_state.gemini_model = None
            logger.error(error_msg)
            log_handler.flush()
            return

        # Configure and initialize
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        # Simple test removed for faster startup, assuming configure/init is enough

        st.session_state.gemini_model = model
        st.session_state.gemini_init_status = "success"
        success_msg = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        st.session_state.gemini_main_status_message = success_msg # Set main message
        logger.info("Gemini Model Initialized Successfully!")

    except Exception as e:
        detailed_error = traceback.format_exc()
        error_message = f"🛑 **Error Initializing Gemini:** {str(e)}" # Simpler message for main screen
        st.session_state.gemini_init_status = "error"
        st.session_state.gemini_main_status_message = error_message # Set main message
        st.session_state.gemini_model = None
        # Log the detailed error
        logger.error(f"Gemini Initialization Error:\n{detailed_error}", exc_info=False)
        log_handler.flush()

# --- Run Initialization ---
init_session_state()
if st.session_state.gemini_init_status == "pending":
    initialize_gemini() # This will set gemini_main_status_message

# --- Core Functions --- (MODIFIED)

def get_timestamp_str():
    """Generates a sortable timestamp string."""
    # Increased precision might help avoid rare collisions on fast systems
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# Save to file (Modified for unique names and directory)
def save_tip_to_file(tip_to_save: str) -> Tuple[bool, str]:
    if not tip_to_save or not isinstance(tip_to_save, str) or tip_to_save.startswith(("Error:", "Sorry,")):
        msg = "⚠️ Invalid tip provided. Cannot save."
        st.warning(msg)
        return False, msg
    try:
        # Ensure directory exists
        os.makedirs(SAVED_TIPS_DIR, exist_ok=True)

        # Generate unique filename
        timestamp = get_timestamp_str()
        filename = f"tip_{timestamp}.txt"
        full_tip_path = os.path.join(SAVED_TIPS_DIR, filename)

        # Save the file
        with open(full_tip_path, "w", encoding='utf-8') as file:
            file.write(tip_to_save)

        msg = f"✅ Tip saved successfully as '{filename}'" # Use new filename
        logger.info(f"Saved tip to: {full_tip_path}")
        return True, msg
    except Exception as e:
        msg = f"🛑 Error saving tip to file: {e}"
        st.error(msg)
        logger.error(f"Save Tip Error", exc_info=True)
        log_handler.flush()
        return False, msg

# Generate audio (Modified for unique names and directory)
def generate_tip_audio(tip_text: str) -> Tuple[bool, Optional[str]]:
    if not tip_text or not isinstance(tip_text, str) or tip_text.startswith(("Error:", "Sorry,")):
        st.warning("⚠️ Invalid text provided. Cannot generate audio.")
        return False, None
    text_to_speak = f"Here's a water saving tip for you: {tip_text}" # Slightly more conversational
    try:
        # Ensure directory exists
        os.makedirs(SAVED_AUDIO_DIR, exist_ok=True)

        # Generate unique filename
        timestamp = get_timestamp_str()
        filename = f"audio_{timestamp}.mp3"
        full_audio_path = os.path.join(SAVED_AUDIO_DIR, filename)

        # REMOVE old file removal logic - no longer needed

        # Generate and save the audio
        tts = gTTS(text=text_to_speak, lang='en', slow=False)
        tts.save(full_audio_path) # Save using full path

        logger.info(f"Successfully saved new audio file: {full_audio_path}")
        # Return the full path for potential immediate use (like the player)
        return True, full_audio_path
    except Exception as e:
        st.error(f"🛑 Text-to-Speech Error: {e}")
        logger.error(f"TTS Error", exc_info=True)
        log_handler.flush()
        return False, None

# --- Streamlit UI Layout ---

# --- Sidebar --- (Shows more detailed status - No change)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3779/3779161.png", width=80) # Simple water drop icon
    st.header("Assistant Status")
    st.markdown("---")

    # Display Detailed Gemini Initialization Status in Sidebar
    st.subheader("🤖 Gemini Model")
    if st.session_state.gemini_init_status == "success":
        st.success(f"✅ **Success!** Model (`{MODEL_NAME}`) loaded.")
        try:
            st.caption(f"google-generativeai version: `{genai.__version__}`")
        except Exception:
             st.caption("Could not get library version.") # Graceful fallback
    elif st.session_state.gemini_init_status == "error":
        # Display the potentially longer error message from init function here
        error_msg_detail = st.session_state.get('gemini_main_status_message', "🛑 **Error:** Initialization failed.")
        st.error(error_msg_detail + "\n\nCheck logs (`app_error.log`) for more details if needed.")
    else: # Pending
        st.info("⏳ Initializing Gemini Model...")

    st.markdown("---")
    st.subheader("🔊 Volume Control")
    st.info("Please use your computer's system volume controls to adjust the audio playback level.")
    st.markdown("---")
    st.caption("App logs errors to `app_error.log`")


# --- Main Page ---
st.title("💧 Water Conservation Assistant")
st.markdown("Let AI help you find ways to save water! Enter a topic or question below.")

# --- Display Initialization Status on Main Screen --- (No change)
# Use the message set during initialization
if 'gemini_main_status_message' in st.session_state and st.session_state.gemini_main_status_message:
    if st.session_state.gemini_init_status == "success":
        st.success(st.session_state.gemini_main_status_message) # Green banner for success
    elif st.session_state.gemini_init_status == "error":
        st.error(st.session_state.gemini_main_status_message) # Red banner for error
    # No need for 'pending' message here, as it happens quickly

st.markdown("---") # Separator after status message

# --- Input Column --- (No change)
col1, col2 = st.columns([2, 3]) # Give more space to the output column

with col1:
    st.subheader("✍️ Generate a Custom Tip")
    st.markdown("What do you need a water-saving tip about?")

    user_prompt = st.text_area(
        "Enter your request (e.g., 'showering', 'washing clothes', 'gardening'):",
        placeholder="Tip for washing dishes efficiently...",
        height=150,
        key="user_ai_prompt",
        label_visibility="collapsed" # Hide label as it's in markdown above
    )

    # Disable button if Gemini isn't ready
    gen_button_disabled = st.session_state.gemini_init_status != "success"
    gen_tooltip = "Gemini model not loaded successfully. Check status messages." if gen_button_disabled else "Generate a water-saving tip based on your input."

    if st.button("✨ Generate Tip using AI", key="generate_ai", disabled=gen_button_disabled, help=gen_tooltip, use_container_width=True):
        st.session_state.current_tip = "" # Clear previous tip
        st.session_state.audio_file_path = None # Clear previous audio playback path
        if user_prompt:
            st.session_state.processing_message = "⏳ Generating AI tip..."
            st.session_state.direct_ai_prompt = user_prompt # Store the prompt
            st.rerun() # Trigger processing block below
        else:
            st.warning("⚠️ Please enter a request or topic for the tip.")

    if gen_button_disabled and st.session_state.gemini_init_status != 'pending': # Only show warning if init failed (not pending)
        st.warning("AI generation is disabled because the Gemini model could not be initialized. Please check the status messages above and in the sidebar for details.", icon="🤖")


# --- Processing Logic Block (Handles ONLY Direct AI Generation) --- (No change)
# (Identical to previous version - no changes needed here)
if st.session_state.processing_message:
    # Display the processing message temporarily
    message_placeholder = st.empty()
    message_placeholder.info(st.session_state.processing_message, icon="⏳")

    # --- Direct AI Generation Logic ---
    if "Generating AI tip" in st.session_state.processing_message:
        model = st.session_state.gemini_model # Get model from state
        if not model:
             st.error("🛑 Direct AI generation failed: Model not available in session state.")
             st.session_state.processing_message = ""
             message_placeholder.empty()
             st.rerun()
        else:
            try:
                prompt_llm = st.session_state.get('direct_ai_prompt', '')
                full_prompt = f"Generate a concise and actionable water conservation tip related to: {prompt_llm}. Focus on practical advice for home use."

                with st.spinner(f"Asking Gemini about '{prompt_llm}'..."):
                    response = model.generate_content(
                        full_prompt,
                         generation_config=genai.types.GenerationConfig(
                            max_output_tokens=150,
                            temperature=0.7
                        )
                    )

                if response and response.candidates and response.text:
                    generated_text = response.text.strip()
                    st.session_state.current_tip = generated_text
                    logger.info(f"Generated tip for '{prompt_llm}': {generated_text}")
                else:
                    feedback_text = "Unknown reason"
                    try:
                        if response and response.prompt_feedback:
                             feedback_text = f"Prompt Feedback: {response.prompt_feedback}"
                        elif response and response.candidates and response.candidates[0].finish_reason:
                             feedback_text = f"Finish Reason: {response.candidates[0].finish_reason.name}"
                             if response.candidates[0].safety_ratings:
                                 feedback_text += f", Safety Ratings: {response.candidates[0].safety_ratings}"
                    except Exception:
                         feedback_text = "Could not parse detailed feedback."

                    st.warning(f"⚠️ AI response was empty or blocked. {feedback_text}", icon="🤖")
                    st.session_state.current_tip = f"Error: Could not generate a tip for '{prompt_llm}'. The AI might have considered the request unsafe or couldn't produce a valid response. Reason: {feedback_text}"
                    logger.warning(f"AI response empty/blocked for '{prompt_llm}'. Feedback: {feedback_text}")

            except Exception as e:
                detailed_error = traceback.format_exc()
                st.error(f"🛑 Error during AI tip generation: {e}")
                logger.error(f"Direct AI Generation Error for prompt '{prompt_llm}':\n{detailed_error}", exc_info=False)
                log_handler.flush()
                st.session_state.current_tip = "Error: An exception occurred while trying to generate the tip. Please check the logs."
            finally:
                st.session_state.processing_message = ""
                message_placeholder.empty()
                st.rerun()


# --- Output Display Column --- (Modified to remove persistent display)
with col2:
    st.subheader("💡 Your Water-Saving Tip")
    st.markdown("---")

    if not st.session_state.processing_message:
        tip_display_area = st.container()

        with tip_display_area:
            if st.session_state.current_tip:
                st.markdown(st.session_state.current_tip) # Display current AI tip
                st.markdown("---")

                # Action Buttons (No changes to button logic itself)
                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    save_disabled = st.session_state.current_tip.startswith("Error:")
                    if st.button("💾 Save Tip", key="save_tip", help="Save the current tip to a new file.", disabled=save_disabled, use_container_width=True):
                        success, msg = save_tip_to_file(st.session_state.current_tip)
                        if success:
                            st.success(msg, icon="💾") # Show success, no rerun needed for this page

                with action_col2:
                    audio_disabled = st.session_state.current_tip.startswith("Error:")
                    if st.button("🔊 Generate Audio", key="generate_audio", help="Generate an audio reading to a new file.", disabled=audio_disabled, use_container_width=True):
                        if st.session_state.current_tip and not audio_disabled:
                            with st.spinner("Generating audio..."):
                                success, audio_path = generate_tip_audio(st.session_state.current_tip)
                                if success:
                                     # Store the full path of the *newly generated* audio
                                     st.session_state.audio_file_path = audio_path
                                     # Use os.path.basename to show just the filename in the message
                                     st.success(f"Audio generated as '{os.path.basename(audio_path)}'!", icon="🔊")
                                     time.sleep(0.1)
                                     st.rerun() # Rerun ONLY to show the player for the new audio immediately

                # --- Display Audio Player ONLY for the *just-generated* audio ---
                # This section now ONLY shows the audio player if one was just generated
                # Persistent audio history will be on the other page
                audio_display_path = st.session_state.get('audio_file_path', None) # Get path from state
                if audio_display_path and os.path.exists(audio_display_path):
                    st.markdown("---") # Separator before the player
                    st.caption("🎧 Playback for newly generated audio:")
                    try:
                        with open(audio_display_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                    except Exception as e:
                        st.error(f"🛑 Error displaying audio player: {e}")
                        logger.error(f"Audio Player Display Error", exc_info=True); log_handler.flush()

                # REMOVED the expander/persistent audio display logic from this main page

            else:
                 st.info("Enter a topic on the left and click 'Generate Tip' to get started!", icon="👈")

# --- Footer --- (No change)
st.markdown("---")
st.caption(f"""
    © {CURRENT_YEAR} Water Saver Assistant | Made with ❤️ for Water Conservation | Designed by **Abdullah F. Al-Shehabi**
""")