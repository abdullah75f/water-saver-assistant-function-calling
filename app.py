# app.py (Modified for multi-file saving AND per-session storage)

import streamlit as st
import os
# requests kept for potential future use
import requests
# random kept for potential future use
import random
import google.generativeai as genai
import datetime # Import datetime to get current year for copyright
from gtts import gTTS
import time # For potential delays
from typing import Union, List, Optional, Tuple, Dict, Any # Add Dict, Any
import traceback # Import traceback for better error logging
import logging # Import standard logging
import sys # To help ensure flushing
import io # Import io for handling bytes in memory

# --- Setup Logging ---
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
MODEL_NAME = 'gemini-1.5-flash-latest' # Define model name centrally
CURRENT_YEAR = datetime.datetime.now().year # Get current year for copyright

# --- REMOVE Directory Constants for Saved Content ---
# APP_DIR = os.path.dirname(__file__)
# SAVED_TIPS_DIR_NAME = "saved_tips"
# SAVED_AUDIO_DIR_NAME = "saved_audio"
# SAVED_TIPS_DIR = os.path.join(APP_DIR, SAVED_TIPS_DIR_NAME)
# SAVED_AUDIO_DIR = os.path.join(APP_DIR, SAVED_AUDIO_DIR_NAME)
# os.makedirs(SAVED_TIPS_DIR, exist_ok=True) # No longer needed here
# os.makedirs(SAVED_AUDIO_DIR, exist_ok=True) # No longer needed here


# --- State Initialization ---
def init_session_state():
    if 'current_tip' not in st.session_state: st.session_state.current_tip = ""
    # 'audio_file_path' is no longer needed for history, use bytes instead
    # if 'audio_file_path' not in st.session_state: st.session_state.audio_file_path = None
    if 'current_audio_bytes' not in st.session_state: st.session_state.current_audio_bytes = None # For immediate playback
    if 'processing_message' not in st.session_state: st.session_state.processing_message = ""
    if 'direct_ai_prompt' not in st.session_state: st.session_state.direct_ai_prompt = ""
    if 'gemini_model' not in st.session_state: st.session_state.gemini_model = None
    if 'gemini_init_status' not in st.session_state: st.session_state.gemini_init_status = "pending" # pending, success, error
    if 'gemini_main_status_message' not in st.session_state: st.session_state.gemini_main_status_message = None

    # --- ADD Session State Lists for History ---
    if 'session_saved_tips' not in st.session_state: st.session_state.session_saved_tips = [] # List of dicts {'filename': str, 'content': str, 'timestamp': datetime}
    if 'session_saved_audio' not in st.session_state: st.session_state.session_saved_audio = [] # List of dicts {'filename': str, 'audio_bytes': bytes, 'timestamp': datetime}

# --- Gemini Model Initialization Function ---
# (No changes needed in initialize_gemini function itself)
def initialize_gemini():
    if st.session_state.gemini_init_status == "success":
        st.session_state.gemini_main_status_message = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        return

    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        # if not api_key:
        #     api_key = os.environ.get("GEMINI_API_KEY") # Optional fallback

        if not api_key:
            st.session_state.gemini_init_status = "error"
            error_msg = "🛑 **Error:** `GEMINI_API_KEY` not found. Please configure it in Streamlit secrets."
            st.session_state.gemini_main_status_message = error_msg
            st.session_state.gemini_model = None
            logger.error(error_msg)
            log_handler.flush()
            return

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        st.session_state.gemini_model = model
        st.session_state.gemini_init_status = "success"
        success_msg = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        st.session_state.gemini_main_status_message = success_msg
        logger.info("Gemini Model Initialized Successfully!")

    except Exception as e:
        detailed_error = traceback.format_exc()
        error_message = f"🛑 **Error Initializing Gemini:** {str(e)}"
        st.session_state.gemini_init_status = "error"
        st.session_state.gemini_main_status_message = error_message
        st.session_state.gemini_model = None
        logger.error(f"Gemini Initialization Error:\n{detailed_error}", exc_info=False)
        log_handler.flush()

# --- Run Initialization ---
init_session_state()
if st.session_state.gemini_init_status == "pending":
    initialize_gemini()

# --- Core Functions --- (MODIFIED FOR SESSION STATE)

def get_timestamp_dt():
    """Gets the current timestamp as a datetime object."""
    return datetime.datetime.now()

def get_timestamp_str(dt_obj: Optional[datetime.datetime] = None) -> str:
    """Generates a sortable timestamp string from a datetime object or now."""
    if dt_obj is None:
        dt_obj = get_timestamp_dt()
    # Increased precision might help avoid rare collisions on fast systems
    return dt_obj.strftime("%Y%m%d_%H%M%S_%f")

# Save tip TO SESSION STATE
def add_tip_to_session(tip_to_save: str) -> Tuple[bool, str]:
    """Adds the given tip to the session state history list."""
    if not tip_to_save or not isinstance(tip_to_save, str) or tip_to_save.startswith(("Error:", "Sorry,")):
        msg = "⚠️ Invalid tip provided. Cannot save to session history."
        st.warning(msg)
        return False, msg
    try:
        # Generate unique filename FOR DISPLAY PURPOSES ONLY
        timestamp_dt = get_timestamp_dt()
        timestamp_str = get_timestamp_str(timestamp_dt)
        filename = f"tip_{timestamp_str}.txt"

        # Create the dictionary entry
        tip_entry = {
            "filename": filename,
            "content": tip_to_save,
            "timestamp": timestamp_dt
        }

        # Append to session state list (insert at beginning for newest first)
        st.session_state.session_saved_tips.insert(0, tip_entry)

        msg = f"✅ Tip added to session history as '{filename}'"
        logger.info(f"Added tip to session state: {filename}")
        return True, msg
    except Exception as e:
        msg = f"🛑 Error adding tip to session state: {e}"
        st.error(msg)
        logger.error(f"Add Tip to Session Error", exc_info=True)
        log_handler.flush()
        return False, msg

# Generate audio and save BYTES TO SESSION STATE, return bytes for immediate playback
def generate_tip_audio(tip_text: str) -> Tuple[bool, Optional[bytes]]:
    """
    Generates audio bytes for the tip, adds them to session state history,
    and returns the bytes for immediate playback.
    """
    if not tip_text or not isinstance(tip_text, str) or tip_text.startswith(("Error:", "Sorry,")):
        st.warning("⚠️ Invalid text provided. Cannot generate audio.")
        return False, None
    text_to_speak = f"Here's a water saving tip for you: {tip_text}"
    try:
        # Generate unique filename FOR DISPLAY PURPOSES ONLY
        timestamp_dt = get_timestamp_dt()
        timestamp_str = get_timestamp_str(timestamp_dt)
        filename = f"audio_{timestamp_str}.mp3"

        # Generate audio into memory
        tts = gTTS(text=text_to_speak, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0) # Rewind the buffer
        audio_bytes = audio_fp.read()

        # Add to session state history (insert at beginning for newest first)
        audio_entry = {
            "filename": filename,
            "audio_bytes": audio_bytes,
            "timestamp": timestamp_dt
            # Optionally store original tip text too:
            # "original_tip": tip_text
        }
        st.session_state.session_saved_audio.insert(0, audio_entry)

        logger.info(f"Successfully generated and added audio to session state: {filename}")
        # Return success and the audio bytes for immediate playback
        return True, audio_bytes
    except Exception as e:
        st.error(f"🛑 Text-to-Speech Error: {e}")
        logger.error(f"TTS Error", exc_info=True)
        log_handler.flush()
        return False, None

# --- Streamlit UI Layout ---

# --- Sidebar --- (No change needed here)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3779/3779161.png", width=80)
    st.header("Assistant Status")
    st.markdown("---")
    st.subheader("🤖 Gemini Model")
    if st.session_state.gemini_init_status == "success":
        st.success(f"✅ **Success!** Model (`{MODEL_NAME}`) loaded.")
        try:
            st.caption(f"google-generativeai version: `{genai.__version__}`")
        except Exception:
             st.caption("Could not get library version.")
    elif st.session_state.gemini_init_status == "error":
        error_msg_detail = st.session_state.get('gemini_main_status_message', "🛑 **Error:** Initialization failed.")
        st.error(error_msg_detail + "\n\nCheck logs (`app_error.log`) for details.")
    else: # Pending
        st.info("⏳ Initializing Gemini Model...")

    st.markdown("---")
    st.subheader("🔊 Volume Control")
    st.info("Use your computer's volume controls.")
    st.markdown("---")
    st.caption("App logs errors to `app_error.log`")
    st.caption(f"Tips/Audio saved in this session: {len(st.session_state.get('session_saved_tips',[]))}/{len(st.session_state.get('session_saved_audio',[]))}")


# --- Main Page ---
st.title("💧 Water Conservation Assistant")
st.markdown("Let AI help you find ways to save water! Enter a topic or question below.")

# Display Initialization Status
if 'gemini_main_status_message' in st.session_state and st.session_state.gemini_main_status_message:
    if st.session_state.gemini_init_status == "success":
        st.success(st.session_state.gemini_main_status_message)
    elif st.session_state.gemini_init_status == "error":
        st.error(st.session_state.gemini_main_status_message)

st.markdown("---")

# --- Input Column ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("✍️ Generate a Custom Tip")
    st.markdown("What do you need a water-saving tip about?")

    user_prompt = st.text_area(
        "Enter your request (e.g., 'showering', 'washing clothes', 'gardening'):",
        placeholder="Tip for washing dishes efficiently...",
        height=150,
        key="user_ai_prompt",
        label_visibility="collapsed"
    )

    gen_button_disabled = st.session_state.gemini_init_status != "success"
    gen_tooltip = "Gemini model not loaded successfully." if gen_button_disabled else "Generate a water-saving tip based on your input."

    if st.button("✨ Generate Tip using AI", key="generate_ai", disabled=gen_button_disabled, help=gen_tooltip, use_container_width=True):
        st.session_state.current_tip = "" # Clear previous tip
        st.session_state.current_audio_bytes = None # Clear previous audio playback bytes
        if user_prompt:
            st.session_state.processing_message = "⏳ Generating AI tip..."
            st.session_state.direct_ai_prompt = user_prompt
            st.rerun()
        else:
            st.warning("⚠️ Please enter a request or topic for the tip.")

    if gen_button_disabled and st.session_state.gemini_init_status != 'pending':
        st.warning("AI generation is disabled (Gemini model init failed).", icon="🤖")


# --- Processing Logic Block (Handles ONLY Direct AI Generation) ---
# (No changes needed in this block - it sets st.session_state.current_tip)
if st.session_state.processing_message:
    message_placeholder = st.empty()
    message_placeholder.info(st.session_state.processing_message, icon="⏳")

    if "Generating AI tip" in st.session_state.processing_message:
        model = st.session_state.gemini_model
        if not model:
             st.error("🛑 Direct AI generation failed: Model not available.")
             st.session_state.processing_message = ""
             message_placeholder.empty()
             # No rerun needed, just stop processing
        else:
            try:
                prompt_llm = st.session_state.get('direct_ai_prompt', '')
                full_prompt = f"Generate a concise and actionable water conservation tip related to: {prompt_llm}. Focus on practical advice for home use. Make it easy to understand."

                with st.spinner(f"Asking Gemini about '{prompt_llm}'..."):
                    response = model.generate_content(
                        full_prompt,
                         generation_config=genai.types.GenerationConfig(
                            max_output_tokens=200, # Slightly more allowance
                            temperature=0.75 # Slightly more creative
                        )
                        # Safety settings can be adjusted if needed
                        # safety_settings={'HARASSMENT':'block_none'}
                    )

                # Robust check for response content
                generated_text = ""
                feedback_text = "Unknown reason"
                finish_reason = "Unknown"
                safety_ratings = "Not available"

                if response:
                    try:
                        generated_text = response.text.strip()
                    except ValueError: # Handle case where .text might raise error (e.g. blocked prompt)
                        generated_text = ""
                        logger.warning(f"Could not access response.text directly for prompt '{prompt_llm}'. Checking feedback.")
                    except Exception as e:
                        generated_text = ""
                        logger.error(f"Unexpected error accessing response.text for '{prompt_llm}': {e}")


                    try:
                        if response.prompt_feedback:
                             feedback_text = f"Prompt Feedback: {response.prompt_feedback}"
                    except Exception: pass # Ignore if prompt_feedback doesn't exist

                    try:
                        if response.candidates:
                             candidate = response.candidates[0]
                             if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                                 # Access enum value if available, otherwise convert to string
                                 finish_reason_val = getattr(candidate.finish_reason, 'name', str(candidate.finish_reason))
                                 finish_reason = f"Finish Reason: {finish_reason_val}"
                             if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                 safety_ratings = f"Safety Ratings: {candidate.safety_ratings}"
                    except Exception: pass # Ignore if candidates structure is unexpected

                # Decide final tip or error message
                if generated_text:
                    st.session_state.current_tip = generated_text
                    logger.info(f"Generated tip for '{prompt_llm}': {generated_text}")
                else:
                    st.warning(f"⚠️ AI response was empty or blocked. {finish_reason}. {feedback_text}. {safety_ratings}", icon="🤖")
                    st.session_state.current_tip = f"Error: Could not generate a tip for '{prompt_llm}'. The AI might have considered the request unsafe or couldn't produce a valid response. {finish_reason}. {feedback_text}."
                    logger.warning(f"AI response empty/blocked for '{prompt_llm}'. {finish_reason}. {feedback_text}. {safety_ratings}")

            except Exception as e:
                detailed_error = traceback.format_exc()
                st.error(f"🛑 Error during AI tip generation: {e}")
                logger.error(f"Direct AI Generation Error for prompt '{prompt_llm}':\n{detailed_error}", exc_info=False)
                log_handler.flush()
                st.session_state.current_tip = "Error: An exception occurred while trying to generate the tip. Please check the logs."
            finally:
                st.session_state.processing_message = ""
                message_placeholder.empty()
                st.rerun() # Rerun to display the result


# --- Output Display Column --- (Modified for session state actions & byte playback)
with col2:
    st.subheader("💡 Your Water-Saving Tip")
    st.markdown("---")

    if not st.session_state.processing_message:
        tip_display_area = st.container()

        with tip_display_area:
            current_tip = st.session_state.get('current_tip')
            if current_tip:
                st.markdown(current_tip) # Display current AI tip
                st.markdown("---")

                # Action Buttons
                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    save_disabled = current_tip.startswith("Error:")
                    # UPDATE BUTTON TEXT/HELP
                    if st.button("💾 Add Tip to History", key="save_tip", help="Add this tip to the current session's history.", disabled=save_disabled, use_container_width=True):
                        # CALL SESSION STATE FUNCTION
                        success, msg = add_tip_to_session(current_tip)
                        if success:
                            st.success(msg, icon="💾") # Show success, no rerun needed just for adding to list
                        else:
                            st.error(msg) # Show error if adding failed

                with action_col2:
                    audio_disabled = current_tip.startswith("Error:")
                    # UPDATE BUTTON TEXT/HELP
                    if st.button("🔊 Generate & Play Audio", key="generate_audio", help="Generate audio for this tip, play it, and add to session history.", disabled=audio_disabled, use_container_width=True):
                        if current_tip and not audio_disabled:
                            with st.spinner("Generating audio..."):
                                # CALL MODIFIED FUNCTION
                                success, audio_bytes_result = generate_tip_audio(current_tip)
                                if success and audio_bytes_result:
                                     # Store bytes for immediate playback below
                                     st.session_state.current_audio_bytes = audio_bytes_result
                                     # Get the filename added to history for the message
                                     # The latest added audio is the first element in the list
                                     if st.session_state.session_saved_audio:
                                         last_audio_filename = st.session_state.session_saved_audio[0]['filename']
                                         st.success(f"Audio generated and added to history as '{last_audio_filename}'!", icon="🔊")
                                     else:
                                         st.success("Audio generated and added to history!", icon="🔊") # Fallback message
                                     # Slight delay might help ensure state update before potential rerun if needed
                                     time.sleep(0.1)
                                     # Rerun needed to display the audio player below using current_audio_bytes
                                     st.rerun()
                                else:
                                     st.error("Audio generation failed.")
                                     st.session_state.current_audio_bytes = None # Clear if failed
                                     st.rerun() # Rerun to clear any old player


                # --- Display Audio Player ONLY for the *just-generated* audio BYTES ---
                # This section now uses the bytes stored in session state
                audio_bytes_to_play = st.session_state.get('current_audio_bytes', None)
                if audio_bytes_to_play:
                    st.markdown("---")
                    st.caption("🎧 Playback for newly generated audio:")
                    try:
                        st.audio(audio_bytes_to_play, format='audio/mp3')
                    except Exception as e:
                        st.error(f"🛑 Error displaying audio player: {e}")
                        logger.error(f"Audio Player Display Error from Bytes", exc_info=True); log_handler.flush()
                # No persistent history display on this page

            else:
                 st.info("Enter a topic on the left and click 'Generate Tip' to get started!", icon="👈")

# --- Footer --- (No change)
st.markdown("---")
st.caption(f"""
    © {CURRENT_YEAR} Water Saver Assistant | Made with ❤️ for Water Conservation | Designed by **Abdullah F. Al-Shehabi**
""")