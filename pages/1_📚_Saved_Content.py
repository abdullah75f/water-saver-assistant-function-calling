# pages/1_📚_Saved_Content.py (Modified for multi-file display)

import streamlit as st
import os
import logging
import datetime # Needed for potential sorting later

# --- Use path relative to *this* script file ---
PAGE_DIR = os.path.dirname(__file__)
# Go one level up from pages/ to find the directory containing app.py and saved folders
APP_ROOT_DIR = os.path.dirname(PAGE_DIR)
# --- Use path relative to *this* script file ---

# --- Define Directory Constants relative to APP_ROOT_DIR ---
SAVED_TIPS_DIR_NAME = "saved_tips"
SAVED_AUDIO_DIR_NAME = "saved_audio"
SAVED_TIPS_DIR = os.path.join(APP_ROOT_DIR, SAVED_TIPS_DIR_NAME)
SAVED_AUDIO_DIR = os.path.join(APP_ROOT_DIR, SAVED_AUDIO_DIR_NAME)

# --- REMOVE Single Filename Constants ---
# TIP_FILENAME = "water_conservation_tip_st.txt"
# AUDIO_FILENAME = "water_tip_audio_st.mp3"

# Basic logger setup for this page if needed
log_handler_page = logging.FileHandler('page_error.log', mode='a', encoding='utf-8') # Consider different log file if desired
log_formatter_page = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler_page.setFormatter(log_formatter_page)
logger_page = logging.getLogger(__name__) # Use __name__ for page-specific logger
logger_page.setLevel(logging.ERROR)
if logger_page.hasHandlers(): logger_page.handlers.clear()
logger_page.addHandler(log_handler_page)


st.set_page_config(
    page_title="Saved Content History", # Updated title
    page_icon="💾"
)

st.title("💾 Saved Tips and Audio History")
st.markdown("---")
st.markdown("This page displays all previously saved tips and generated audio files, sorted newest first.")

# --- Display List of Saved Tips ---
st.subheader("📄 Saved Tips History")
try:
    # Ensure directory exists before listing
    if os.path.exists(SAVED_TIPS_DIR) and os.path.isdir(SAVED_TIPS_DIR):
        tip_files = sorted(
            # Filter for files starting with 'tip_' and ending with '.txt'
            [f for f in os.listdir(SAVED_TIPS_DIR) if f.startswith("tip_") and f.endswith(".txt") and os.path.isfile(os.path.join(SAVED_TIPS_DIR, f))],
            reverse=True # Show newest first (based on timestamp filename)
        )
        if not tip_files:
            st.info("No saved tips found in the `saved_tips` directory yet.")
        else:
            st.caption(f"Found {len(tip_files)} saved tip file(s):")
            for filename in tip_files:
                full_tip_path = os.path.join(SAVED_TIPS_DIR, filename)
                try:
                    # Use an expander for each tip, showing the filename
                    with st.expander(f"View Tip: `{filename}`"):
                        with open(full_tip_path, "r", encoding='utf-8') as f:
                            saved_content = f.read()
                        st.text(saved_content) # Display content as plain text
                except Exception as e:
                    st.error(f"Error reading tip file `{filename}`: {e}")
                    logger_page.error(f"Error reading {full_tip_path}", exc_info=True)
                    log_handler_page.flush()
    else:
        st.info("The directory for saved tips (`saved_tips/`) doesn't exist yet. Save a tip on the main page first.")
        logger_page.info(f"Checked for tips directory at: {SAVED_TIPS_DIR}")

except Exception as e:
    st.error(f"An error occurred while listing saved tips: {e}")
    logger_page.error("Error listing tips directory", exc_info=True); log_handler_page.flush()


st.markdown("---") # Separator

# --- Display List of Saved Audio ---
st.subheader("🎧 Saved Audio History")
try:
    # Ensure directory exists before listing
    if os.path.exists(SAVED_AUDIO_DIR) and os.path.isdir(SAVED_AUDIO_DIR):
        audio_files = sorted(
            # Filter for files starting with 'audio_' and ending with '.mp3'
            [f for f in os.listdir(SAVED_AUDIO_DIR) if f.startswith("audio_") and f.endswith(".mp3") and os.path.isfile(os.path.join(SAVED_AUDIO_DIR, f))],
            reverse=True # Show newest first
        )
        if not audio_files:
            st.info("No saved audio found in the `saved_audio` directory yet.")
        else:
            st.caption(f"Found {len(audio_files)} saved audio file(s):")
            for filename in audio_files:
                 full_audio_path = os.path.join(SAVED_AUDIO_DIR, filename)
                 try:
                     st.markdown(f"**File:** `{filename}`") # Display filename clearly above player
                     with open(full_audio_path, "rb") as audio_file:
                         audio_bytes = audio_file.read()
                     st.audio(audio_bytes, format='audio/mp3')
                     st.markdown("---") # Separator between different audio files
                 except Exception as e:
                     st.error(f"🛑 Error displaying audio player for `{filename}`: {e}")
                     logger_page.error(f"Audio Player Display Error for {full_audio_path}", exc_info=True)
                     log_handler_page.flush()
    else:
        st.info("The directory for saved audio (`saved_audio/`) doesn't exist yet. Generate audio on the main page first.")
        logger_page.info(f"Checked for audio directory at: {SAVED_AUDIO_DIR}")

except Exception as e:
    st.error(f"An error occurred while listing saved audio: {e}")
    logger_page.error("Error listing audio directory", exc_info=True); log_handler_page.flush()


st.markdown("---")
st.caption("This page reads directly from the `saved_tips` and `saved_audio` directories.")