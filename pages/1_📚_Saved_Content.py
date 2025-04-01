import streamlit as st
import logging
import datetime 

log_handler_page = logging.FileHandler('page_error.log', mode='a', encoding='utf-8')
log_formatter_page = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler_page.setFormatter(log_formatter_page)
logger_page = logging.getLogger(__name__)
logger_page.setLevel(logging.ERROR)
if logger_page.hasHandlers(): logger_page.handlers.clear()
logger_page.addHandler(log_handler_page)

st.set_page_config(
    page_title="Session Content History", 
    page_icon="💾"
)

st.title("💾 Current Session History") 
st.markdown("---")
st.markdown("This page displays the tips and audio generated *during this browser session*. Content will be lost when you close the browser tab.")

# --- Display List of Saved Tips FROM SESSION STATE ---
st.subheader("📄 Saved Tips (This Session)")
try:
    # Access the list from session state
    saved_tips_list = st.session_state.get('session_saved_tips', [])

    if not saved_tips_list:
        st.info("No tips have been added to the history in this session yet.")
    else:
        st.caption(f"Found {len(saved_tips_list)} tip(s) in this session's history (newest first):")
        # Iterate through the list (already ordered newest first)
        for tip_entry in saved_tips_list:
            filename = tip_entry.get('filename', 'Unknown Filename')
            content = tip_entry.get('content', 'No content found.')
            timestamp = tip_entry.get('timestamp') # Get timestamp if available
            display_ts = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "N/A"

            try:
                # Use an expander for each tip, showing the filename and timestamp
                with st.expander(f"View Tip: `{filename}` (Saved: {display_ts})"):
                    st.text(content) # Display content as plain text
            except Exception as e:
                st.error(f"Error displaying tip entry `{filename}`: {e}")
                logger_page.error(f"Error displaying session tip: {filename}", exc_info=True)
                log_handler_page.flush()

except Exception as e:
    st.error(f"An error occurred while retrieving tips from session state: {e}")
    logger_page.error("Error accessing session_saved_tips", exc_info=True); log_handler_page.flush()


st.markdown("---") 

# --- Display List of Saved Audio FROM SESSION STATE ---
st.subheader("🎧 Saved Audio (This Session)")
try:
    # Access the list from session state
    saved_audio_list = st.session_state.get('session_saved_audio', [])

    if not saved_audio_list:
        st.info("No audio has been generated and saved in this session yet.")
    else:
        st.caption(f"Found {len(saved_audio_list)} audio clip(s) in this session's history (newest first):")
        # Iterate through the list (already ordered newest first)
        for audio_entry in saved_audio_list:
            filename = audio_entry.get('filename', 'Unknown Filename')
            audio_bytes = audio_entry.get('audio_bytes')
            timestamp = audio_entry.get('timestamp')
            display_ts = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "N/A"

            if audio_bytes:
                try:
                    st.markdown(f"**File:** `{filename}` (Generated: {display_ts})") # Display filename clearly above player
                    st.audio(audio_bytes, format='audio/mp3')
                    st.markdown("---") # Separator between different audio files
                except Exception as e:
                    st.error(f"🛑 Error displaying audio player for `{filename}`: {e}")
                    logger_page.error(f"Session Audio Player Display Error for {filename}", exc_info=True)
                    log_handler_page.flush()
            else:
                 st.warning(f"Audio data missing for entry '{filename}'.")

except Exception as e:
    st.error(f"An error occurred while retrieving audio from session state: {e}")
    logger_page.error("Error accessing session_saved_audio", exc_info=True); log_handler_page.flush()


st.markdown("---")
st.caption("This page reads directly from the current browser session's memory (`st.session_state`).")