import streamlit as st
import os
import requests  
import random  
import google.generativeai as genai
import datetime  
from gtts import gTTS
import time  
from typing import Union, List, Optional, Tuple, Dict, Any 
import traceback  
import logging  
import sys  
import io  
import smtplib
from email.message import EmailMessage
import socket  

log_handler = logging.FileHandler('app_error.log', mode='a', encoding='utf-8')
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if not logger.hasHandlers():
    logger.addHandler(log_handler)


# --- Configuration & Constants ---
st.set_page_config(
    page_title="💧 Water Saver Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💧"
)

MODEL_NAME = 'gemini-1.5-flash-latest'  # Define model name centrally
CURRENT_YEAR = datetime.datetime.now().year  # Get current year for copyright

# --- State Initialization ---
# (Initializes all session state variables needed)
def init_session_state():
    # Core app state
    if 'current_tip' not in st.session_state: st.session_state.current_tip = ""
    if 'current_audio_bytes' not in st.session_state: st.session_state.current_audio_bytes = None
    if 'processing_message' not in st.session_state: st.session_state.processing_message = ""
    if 'direct_ai_prompt' not in st.session_state: st.session_state.direct_ai_prompt = ""
    # Gemini state
    if 'gemini_model' not in st.session_state: st.session_state.gemini_model = None
    if 'gemini_init_status' not in st.session_state: st.session_state.gemini_init_status = "pending"
    if 'gemini_main_status_message' not in st.session_state: st.session_state.gemini_main_status_message = None
    # Session history lists
    if 'session_saved_tips' not in st.session_state: st.session_state.session_saved_tips = []
    if 'session_saved_audio' not in st.session_state: st.session_state.session_saved_audio = []
    # Email recipient state
    if 'recipient_email' not in st.session_state: st.session_state.recipient_email = ""

# --- Gemini Model Initialization Function ---
# (Handles initializing the Gemini model using API key from secrets)
def initialize_gemini():
    if st.session_state.gemini_init_status == "success":
        st.session_state.gemini_main_status_message = f"✅ Gemini Model (`{MODEL_NAME}`) Initialized Successfully!"
        return

    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.session_state.gemini_init_status = "error"
            error_msg = "🛑 **Error:** `GEMINI_API_KEY` not found in Streamlit secrets. AI features disabled."
            st.session_state.gemini_main_status_message = error_msg
            st.session_state.gemini_model = None
            logger.error(error_msg); log_handler.flush()
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
        logger.error(f"Gemini Initialization Error:\n{detailed_error}", exc_info=False); log_handler.flush()

# --- Run Initialization ---
init_session_state()
# Initialize Gemini only once if status is pending
if st.session_state.gemini_init_status == "pending":
    initialize_gemini()

# --- Core Utility Functions ---

def get_timestamp_dt():
    """Gets the current timestamp as a datetime object."""
    return datetime.datetime.now()

def get_timestamp_str(dt_obj: Optional[datetime.datetime] = None) -> str:
    """Generates a sortable timestamp string from a datetime object or now."""
    if dt_obj is None: dt_obj = get_timestamp_dt()
    return dt_obj.strftime("%Y%m%d_%H%M%S_%f")

# --- Tip History Function ---
def add_tip_to_session(tip_to_save: str) -> Tuple[bool, str]:
    """Adds the given tip to the session state history list."""
    if not tip_to_save or not isinstance(tip_to_save, str) or tip_to_save.startswith(("Error:", "Sorry,")):
        return False, "⚠️ Invalid tip provided. Cannot save to session history."
    try:
        timestamp_dt = get_timestamp_dt()
        timestamp_str = get_timestamp_str(timestamp_dt)
        filename = f"tip_{timestamp_str}.txt"
        tip_entry = {"filename": filename, "content": tip_to_save, "timestamp": timestamp_dt}
        st.session_state.session_saved_tips.insert(0, tip_entry) # Newest first
        msg = f"✅ Tip added to session history as '{filename}'"
        logger.info(f"Added tip to session state: {filename}")
        return True, msg
    except Exception as e:
        msg = f"🛑 Error adding tip to session state: {e}"
        logger.error(f"Add Tip to Session Error", exc_info=True); log_handler.flush()
        return False, msg

# --- Audio Generation & History Function ---
def generate_tip_audio(tip_text: str) -> Tuple[bool, Optional[bytes]]:
    """Generates audio bytes, adds them to session history, and returns bytes for playback."""
    if not tip_text or not isinstance(tip_text, str) or tip_text.startswith(("Error:", "Sorry,")):
        return False, None
    text_to_speak = f"Here's a water saving tip for you: {tip_text}"
    try:
        timestamp_dt = get_timestamp_dt()
        timestamp_str = get_timestamp_str(timestamp_dt)
        filename = f"audio_{timestamp_str}.mp3"
        tts = gTTS(text=text_to_speak, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()
        audio_entry = {"filename": filename, "audio_bytes": audio_bytes, "timestamp": timestamp_dt}
        st.session_state.session_saved_audio.insert(0, audio_entry) 
        logger.info(f"Generated and added audio to session state: {filename}")
        return True, audio_bytes
    except Exception as e:
        logger.error(f"TTS Error generating audio for tip", exc_info=True); log_handler.flush()
        return False, None 

# --- Email Sending Function ---
def send_tip_email(recipient_email: str, tip_content: str) -> Tuple[bool, str]:
    """Sends the water-saving tip via email using credentials from secrets.toml."""
    # 1. Retrieve Credentials SECURELY from secrets.toml
    try:
        sender_email = st.secrets["EMAIL_SENDER_ADDRESS"]
        sender_password = st.secrets["EMAIL_SENDER_PASSWORD"] 
        smtp_server = st.secrets["SMTP_SERVER"]
        smtp_port = int(st.secrets["SMTP_PORT"])
    except KeyError as e:
        msg = f"🛑 Email Setup Error: Missing '{e}' in secrets.toml. Cannot send email."
        logger.error(msg); log_handler.flush()
        return False, msg
    except (ValueError, TypeError):
         msg = "🛑 Email Setup Error: SMTP_PORT in secrets.toml must be a number."
         logger.error(msg); log_handler.flush()
         return False, msg
    except Exception as e: 
        msg = f"🛑 Email Setup Error: Could not read email configuration: {e}"
        logger.error(msg, exc_info=True); log_handler.flush()
        return False, msg

    # 2. Validate Recipient Email Address (basic check)
    if not recipient_email or "@" not in recipient_email or "." not in recipient_email.split('@')[-1]:
        return False, "⚠️ Please enter a valid recipient email address."

    # 3. Construct the Email Message
    msg = EmailMessage()
    msg['Subject'] = "💧 Your Water Saving Tip from the Assistant!"
    msg['From'] = sender_email
    msg['To'] = recipient_email
    email_body = f"""
Hi there,

Here's the water conservation tip you requested:

"{tip_content}"

Keep saving water!

Best regards,
Your Water Saver Assistant
"""
    msg.set_content(email_body)

    # 4. Send the Email via SMTP
    try:
        logger.info(f"Attempting SMTP connection: {smtp_server}:{smtp_port}")
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server: 
            server.ehlo() 
            server.starttls() 
            server.ehlo() 
            logger.info(f"Attempting login as {sender_email}")
            server.login(sender_email, sender_password)
            logger.info(f"Sending email to {recipient_email}")
            server.send_message(msg)
            logger.info(f"Email successfully sent to {recipient_email}")
        return True, f"✅ Email sent successfully to {recipient_email}!"

    # Specific error handling for common SMTP issues
    except smtplib.SMTPAuthenticationError:
        error_msg = "🛑 Email Sending Failed: Authentication error. Check sender email/App Password in secrets.toml."
        logger.error(error_msg); log_handler.flush()
        return False, error_msg
    except (socket.gaierror, smtplib.SMTPConnectError, ConnectionRefusedError, TimeoutError) as e:
        error_msg = f"🛑 Email Sending Failed: Connection error ({smtp_server}:{smtp_port}). Check server/port and network. Details: {e}"
        logger.error(error_msg); log_handler.flush()
        return False, error_msg
    except smtplib.SMTPException as e: 
        error_msg = f"🛑 Email Sending Failed: An SMTP error occurred: {e}"
        logger.error(error_msg, exc_info=True); log_handler.flush()
        return False, error_msg
    except Exception as e: 
        error_msg = f"🛑 Email Sending Failed: An unexpected error occurred: {e}"
        logger.error(error_msg, exc_info=True); log_handler.flush()
        return False, error_msg


# --- Streamlit UI Layout ---
# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3779/3779161.png", width=80)
    st.header("Assistant Status")
    st.markdown("---")

    # Gemini Status Display
    st.subheader("🤖 Gemini Model")
    if st.session_state.gemini_init_status == "success":
        st.success(f"✅ **Success!** Model (`{MODEL_NAME}`) loaded.")
        try: st.caption(f"google-generativeai version: `{genai.__version__}`")
        except Exception: st.caption("Could not get library version.")
    elif st.session_state.gemini_init_status == "error":
        error_msg_detail = st.session_state.get('gemini_main_status_message', "🛑 **Error:** Initialization failed.")
        st.error(error_msg_detail + "\n\nCheck `app_error.log` for details.")
    else: 
        st.info("⏳ Initializing Gemini Model...")

    st.markdown("---")

    # Email Setup Check (Reads secrets but only displays sender email publicly)
    st.subheader("✉️ Email Setup Check")
    try:
        sender_addr_check = st.secrets["EMAIL_SENDER_ADDRESS"] 
        st.secrets["EMAIL_SENDER_PASSWORD"] 
        st.secrets["SMTP_SERVER"] 
        int(st.secrets["SMTP_PORT"]) 
        st.success("✅ Email secrets appear configured.")
        
        st.caption(f"Sender: {sender_addr_check}")
    except KeyError as e:
        st.warning(f"⚠️ Email secrets missing (`{e}`). Check `.streamlit/secrets.toml`.", icon="📄")
    except (ValueError, TypeError):
        st.warning("⚠️ `SMTP_PORT` in secrets must be a number.", icon="📄")
    except Exception as e:
         st.error(f"🛑 Error checking email secrets: {e}", icon="📄")

    st.markdown("---")
    st.subheader("🔊 Volume Control")
    st.info("Use your computer's volume controls.")
    st.markdown("---")
    st.caption(f"Session History: {len(st.session_state.get('session_saved_tips',[]))} Tips / {len(st.session_state.get('session_saved_audio',[]))} Audio")
    st.caption("App logs errors to `app_error.log`")


# --- Main Page ---
st.title("💧 Water Conservation Assistant")
st.markdown("Let AI help you find ways to save water! Enter a topic or question below.")

# Display Gemini Initialization Status on Main Screen
if 'gemini_main_status_message' in st.session_state and st.session_state.gemini_main_status_message:
    # Show status message only once after initialization attempt
    status_msg = st.session_state.gemini_main_status_message
    if status_msg: 
        if st.session_state.gemini_init_status == "success":
            st.success(status_msg)
        elif st.session_state.gemini_init_status == "error":
            st.error(status_msg)
        

st.markdown("---") 

# --- Input Column (Left) ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("✍️ Generate a Custom Tip")
    st.markdown("What do you need a water-saving tip about?")

    user_prompt = st.text_area(
        "Enter your request (e.g., 'showering', 'washing clothes', 'gardening'):",
        placeholder="Tip for washing dishes efficiently...",
        height=150,
        key="user_ai_prompt", 
        label_visibility="collapsed" # 
    )

    gen_button_disabled = st.session_state.gemini_init_status != "success"
    gen_tooltip = "Gemini model not loaded successfully." if gen_button_disabled else "Generate a water-saving tip."

    if st.button("✨ Generate Tip using AI", key="generate_ai_button", disabled=gen_button_disabled, help=gen_tooltip, use_container_width=True):
        # Reset relevant states for new generation
        st.session_state.current_tip = ""
        st.session_state.current_audio_bytes = None
        st.session_state.recipient_email = "" 

        if user_prompt:
            st.session_state.processing_message = "⏳ Generating AI tip..."
            st.session_state.direct_ai_prompt = user_prompt
            st.rerun() 
        else:
            st.warning("⚠️ Please enter a request or topic first.")

    if gen_button_disabled and st.session_state.gemini_init_status == "error":
        # Show warning only if Gemini failed (not pending)
        st.warning("AI generation is disabled (Gemini model initialization failed).", icon="🤖")


# --- Processing Logic Block ---
if st.session_state.processing_message:
    # Display temporary processing message
    message_placeholder = st.empty()
    message_placeholder.info(st.session_state.processing_message, icon="⏳")

    # --- Direct AI Generation ---
    if "Generating AI tip" in st.session_state.processing_message:
        model = st.session_state.gemini_model
        if not model:
             st.error("🛑 AI Generation Error: Model not available in session state.")
             st.session_state.current_tip = "Error: AI Model not loaded." 
             st.session_state.processing_message = "" 
        else:
            try:
                prompt_llm = st.session_state.get('direct_ai_prompt', '')
                # Construct a clear prompt for the LLM
                full_prompt = f"Generate one concise and actionable water conservation tip related to: '{prompt_llm}'. Focus on practical advice for home use. Make it easy to understand and implement. Start directly with the tip, no introductory phrases like 'Here's a tip:'."

                with st.spinner(f"Asking Gemini about '{prompt_llm}'..."):
                    response = model.generate_content(
                        full_prompt,
                         generation_config=genai.types.GenerationConfig(
                            max_output_tokens=250, 
                            temperature=0.7 
                        ),
                        
                    )

                # Process the response robustly
                generated_text = ""; feedback_text = "Unknown"; finish_reason = "Unknown"; safety_ratings = "N/A"
                if response:
                    # Try getting text, handle potential blocks/errors
                    try: generated_text = response.text.strip()
                    except ValueError: logger.warning(f"Could not access response.text for '{prompt_llm}'. Checking feedback.")
                    except Exception as e: logger.error(f"Unexpected error accessing response.text for '{prompt_llm}': {e}")

                    # Try getting feedback/reasons if text is empty or for logging
                    try:
                        if response.prompt_feedback and response.prompt_feedback.block_reason:
                            feedback_text = f"Blocked due to: {response.prompt_feedback.block_reason}"
                    except Exception: pass 
                    try:
                        if response.candidates:
                             candidate = response.candidates[0]
                             if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                                 finish_reason_val = getattr(candidate.finish_reason, 'name', str(candidate.finish_reason))
                                
                                 if finish_reason_val not in ['STOP', 'MAX_TOKENS']:
                                     finish_reason = f"Finish Reason: {finish_reason_val}"
                             if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                 # Summarize safety ratings if they exist
                                 ratings_summary = ", ".join([f"{r.category.name}: {r.probability.name}" for r in candidate.safety_ratings])
                                 safety_ratings = f"Safety: [{ratings_summary}]"
                    except Exception: pass #

                # Set session state based on whether text was generated
                if generated_text:
                    st.session_state.current_tip = generated_text
                    logger.info(f"Generated tip for '{prompt_llm}'.")
                else:
                    # Construct informative error message
                    err_detail = f"{finish_reason}. {feedback_text}. {safety_ratings}"
                    st.session_state.current_tip = f"Error: Could not generate a tip for '{prompt_llm}'. The AI response was empty or blocked. ({err_detail})"
                    st.warning(f"⚠️ AI response empty/blocked. Details: {err_detail}", icon="🤖") 
                    logger.warning(f"AI response empty/blocked for '{prompt_llm}'. Details: {err_detail}")

            except Exception as e: 
                detailed_error = traceback.format_exc()
                st.error(f"🛑 Error during AI tip generation: {e}") 
                logger.error(f"Direct AI Generation Error for prompt '{prompt_llm}':\n{detailed_error}", exc_info=False)
                log_handler.flush()
                st.session_state.current_tip = "Error: An exception occurred during AI generation. Please check logs."
            finally:
                st.session_state.processing_message = ""
                message_placeholder.empty()
                st.rerun()


# --- Output Display Column (Right) ---
with col2:
    st.subheader("💡 Your Water-Saving Tip")
    st.markdown("---")

    # Display content only if not actively processing generation
    if not st.session_state.processing_message:
        tip_display_area = st.container() 

        with tip_display_area:
            current_tip = st.session_state.get('current_tip')

            # Display the tip if one exists (even error messages)
            if current_tip:
                st.markdown(current_tip) 
                st.markdown("---")

                # --- Action Buttons (History & Audio) ---
                action_col1, action_col2 = st.columns(2)
                # Enable actions only if the current tip is NOT an error message
                tip_is_valid_for_actions = current_tip and not current_tip.startswith("Error:")

                with action_col1:
                    # Button to add tip to session history
                    if st.button("💾 Add Tip to History", key="save_tip_button",
                                 help="Add this tip to the current session's history page.",
                                 disabled=not tip_is_valid_for_actions, use_container_width=True):
                        if tip_is_valid_for_actions:
                            success, msg = add_tip_to_session(current_tip)
                            if success: st.success(msg, icon="💾") 
                            else: st.error(msg) 

                with action_col2:
                    # Button to generate and play audio
                    if st.button("🔊 Generate & Play Audio", key="generate_audio_button",
                                 help="Generate audio, play it here, and add to history.",
                                 disabled=not tip_is_valid_for_actions, use_container_width=True):
                        if tip_is_valid_for_actions:
                            with st.spinner("Generating audio..."):
                                success, audio_bytes_result = generate_tip_audio(current_tip)
                            if success and audio_bytes_result:
                                 st.session_state.current_audio_bytes = audio_bytes_result 
                                 last_audio_filename = "audio clip"
                                 if st.session_state.session_saved_audio: 
                                     last_audio_filename = f"'{st.session_state.session_saved_audio[0]['filename']}'"
                                 st.success(f"Audio generated & added as {last_audio_filename}!", icon="🔊")
                                 time.sleep(0.1) 
                                 st.rerun() 
                            else:
                                 st.error("Audio generation failed. Please try again or check logs.")
                                 st.session_state.current_audio_bytes = None 

                # --- Audio Player ---
                audio_bytes_to_play = st.session_state.get('current_audio_bytes', None)
                if audio_bytes_to_play:
                    st.markdown("---") 
                    st.caption("🎧 Playback for newly generated audio:")
                    try:
                        st.audio(audio_bytes_to_play, format='audio/mp3')
                    except Exception as e:
                        st.error(f"🛑 Error displaying audio player: {e}")
                        logger.error(f"Audio Player Display Error from Bytes", exc_info=True); log_handler.flush()

                # --- Email Section ---
                st.markdown("---") 
                st.subheader("✉️ Send Tip via Email")

                # Check if email secrets are okay (avoids repeating the check logic)
                email_secrets_ok = False
                secrets_error_msg = "" 
                try:
                    st.secrets["EMAIL_SENDER_ADDRESS"] #
                    # st.secrets["EMAIL_SENDER_PASSWORD"]
                    # st.secrets["SMTP_SERVER"]
                    int(st.secrets["SMTP_PORT"])
                    email_secrets_ok = True
                except KeyError as e:
                     secrets_error_msg = f"Email disabled: Missing secret '{e}'."
                except (ValueError, TypeError):
                     secrets_error_msg = "Email disabled: SMTP_PORT must be a number."
                except Exception as e:
                     secrets_error_msg = f"Email disabled: Error checking secrets ({e})."

                # Determine if email UI should be fully disabled
                email_fully_disabled = not tip_is_valid_for_actions or not email_secrets_ok

                # Show warning only if secrets are the reason for disabling
                if not email_secrets_ok:
                     st.warning(f"{secrets_error_msg} Check `.streamlit/secrets.toml` configuration.", icon="🔒")

                # Input field for Recipient's email address
                st.session_state.recipient_email = st.text_input(
                    "Recipient Email:",
                    placeholder="Enter recipient's email address" if not email_fully_disabled else ("Generate a valid tip first" if email_secrets_ok else "Email sending is not configured"),
                    key="recipient_email_input", # Unique key
                    value=st.session_state.recipient_email, # Use session state to preserve input
                    disabled=email_fully_disabled,
                    label_visibility="collapsed" # Use placeholder and subheader instead
                )

                # Determine appropriate help text for the send button
                email_button_help = "Send the current tip via email."
                if not email_secrets_ok:
                    email_button_help = "Email sending is disabled due to missing/invalid configuration in secrets.toml."
                elif not tip_is_valid_for_actions:
                    email_button_help = "Generate a valid tip before sending via email."

                # Button to trigger sending the email
                if st.button("📧 Send Email", key="send_email_button",
                             help=email_button_help, use_container_width=True,
                             disabled=email_fully_disabled):

                    recipient = st.session_state.recipient_email.strip() # Get and strip whitespace
                    if recipient:
                         # Show spinner while sending
                         with st.spinner(f"Sending email to {recipient}..."):
                             success, msg = send_tip_email(recipient, current_tip)
                         # Display result message (success or error)
                         if success:
                             st.success(msg)
                         else:
                             st.error(msg) 
                    else:
                         st.warning("⚠️ Please enter a recipient email address first.")

            elif not st.session_state.processing_message: 
                 st.info("Enter a topic on the left and click 'Generate Tip' to get started, or view history.", icon="👈")

# --- Footer ---
st.markdown("---")
st.caption(f"""
    © {CURRENT_YEAR} Water Saver Assistant | Made with ❤️ for Water Conservation | Designed by **Abdullah F. Al-Shehabi**
""")

