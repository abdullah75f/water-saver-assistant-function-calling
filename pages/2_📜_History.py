# pages/2_📜_History.py
# Streamlit page to display session history (tips and audio)

import streamlit as st
import logging
import datetime

# --- Logging Setup for this Page ---
# Use a separate log file or configure the main logger if preferred
log_handler_page = logging.FileHandler('history_page_error.log', mode='a', encoding='utf-8')
log_formatter_page = logging.Formatter('%(asctime)s - %(levelname)s - Page: History - %(message)s')
log_handler_page.setFormatter(log_formatter_page)
# Get a logger specific to this page/module
logger_page = logging.getLogger(__name__) # Use __name__ for page-specific logger
logger_page.setLevel(logging.ERROR) # Log errors specific to this page display
# Avoid adding handlers multiple times if the page reruns quickly
if not logger_page.hasHandlers():
    logger_page.addHandler(log_handler_page)
logger_page.info("--- History Page Loaded ---") # Log page load

# --- Page Configuration ---
st.set_page_config(
    page_title="Session Content History",
    page_icon="💾",
    layout="wide" # Use wide layout for potentially long lists
)

# --- Page Content ---
st.title("💾 Current Session History")
st.markdown("---")
st.markdown(
    "This page displays the tips and audio generated and explicitly saved *during this browser session* "
    "using the 'save tip' or 'tell tip' actions triggered via the AI assistant. "
    "**Content will be lost when you close or refresh this browser tab.**"
)
st.info(
    "Note: Only items processed by the `save_tip` or `tell_tip` functions called by the AI will appear here. "
    "Tips merely displayed by the AI are not automatically saved to this history.", icon="ℹ️"
)


# --- Display List of Saved Tips FROM SESSION STATE ---
st.subheader("📄 Saved Tips (This Session)")
try:
    # Safely access the list from session state using .get() with a default empty list
    saved_tips_list = st.session_state.get('session_saved_tips', [])

    if not saved_tips_list:
        st.info("No tips have been added to the history in this session yet via the 'save tip' action.")
    else:
        st.caption(f"Found {len(saved_tips_list)} tip(s) in this session's history (newest first):")
        # Iterate through the list (it's already ordered newest first due to insert(0, ...))
        for idx, tip_entry in enumerate(saved_tips_list):
            # Safely get data from the dictionary entry
            filename = tip_entry.get('filename', f'Unknown Filename {idx+1}')
            content = tip_entry.get('content', 'Error: Content missing.')
            timestamp = tip_entry.get('timestamp') # Get the datetime object
            # Format timestamp nicely, handle if it's somehow missing
            display_ts = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime.datetime) else "N/A"

            try:
                # Use an expander for each tip for cleaner display
                with st.expander(f"View Tip: `{filename}` (Saved: {display_ts})"):
                    # Display content using st.text for preformatted or st.markdown if it might have formatting
                    st.text(content)
                    # Provide a download button for the text content
                    st.download_button(
                        label="Download Tip as TXT",
                        data=content,
                        file_name=filename, # Use the generated filename
                        mime="text/plain",
                        key=f"download_tip_{idx}" # Unique key for the download button
                    )
            except Exception as e_display_tip:
                # Log and show error if displaying a specific entry fails
                st.error(f"Error displaying tip entry `{filename}`: {e_display_tip}")
                logger_page.error(f"Error rendering session tip: {filename}", exc_info=True)
                log_handler_page.flush()

except Exception as e_access_tips:
    # Log and show error if accessing the session state list itself fails
    st.error(f"An error occurred while retrieving tips from the session state: {e_access_tips}")
    logger_page.error("Error accessing st.session_state['session_saved_tips']", exc_info=True); log_handler_page.flush()


st.markdown("---") # Separator

# --- Display List of Saved Audio FROM SESSION STATE ---
st.subheader("🎧 Generated Audio (This Session)")
try:
    # Safely access the list from session state
    saved_audio_list = st.session_state.get('session_saved_audio', [])

    if not saved_audio_list:
        st.info("No audio has been generated and saved in this session yet via the 'tell tip' action.")
    else:
        st.caption(f"Found {len(saved_audio_list)} audio clip(s) in this session's history (newest first):")
        # Iterate through the list (newest first)
        for idx, audio_entry in enumerate(saved_audio_list):
            # Safely get data from the dictionary entry
            filename = audio_entry.get('filename', f'Unknown Audio {idx+1}')
            audio_bytes = audio_entry.get('audio_bytes')
            timestamp = audio_entry.get('timestamp')
            # Format timestamp
            display_ts = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime.datetime) else "N/A"

            if audio_bytes and isinstance(audio_bytes, bytes):
                try:
                    # Display filename/timestamp clearly above the player
                    st.markdown(f"**File:** `{filename}` (Generated: {display_ts})")
                    # Display the audio player using the stored bytes
                    st.audio(audio_bytes, format='audio/mp3') # Assume mp3 format from gTTS
                    # Provide download button for the audio
                    st.download_button(
                         label="Download Audio (MP3)",
                         data=audio_bytes,
                         file_name=filename, # Use the generated filename
                         mime="audio/mpeg", # Correct MIME type for MP3
                         key=f"download_audio_{idx}" # Unique key
                    )
                    st.markdown("---") # Separator between audio clips
                except Exception as e_display_audio:
                    # Log and show error if displaying the audio player fails
                    st.error(f"🛑 Error displaying audio player for `{filename}`: {e_display_audio}")
                    logger_page.error(f"Session Audio Player Display Error for {filename}", exc_info=True)
                    log_handler_page.flush()
            else:
                 # Log warning if entry exists but audio data is missing or invalid
                 st.warning(f"Audio data missing or invalid for entry '{filename}'.")
                 logger_page.warning(f"Audio data missing/invalid for session audio history entry: {filename}")

except Exception as e_access_audio:
    # Log and show error if accessing the session state list fails
    st.error(f"An error occurred while retrieving audio history from the session state: {e_access_audio}")
    logger_page.error("Error accessing st.session_state['session_saved_audio']", exc_info=True); log_handler_page.flush()


st.markdown("---")
st.caption("End of session history.")
# Final log flush for the page
log_handler_page.flush()