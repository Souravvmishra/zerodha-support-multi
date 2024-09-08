import os
import time
import google.generativeai as genai
import streamlit as st

# Set up your environment variable for the API key
genai.configure(api_key='AIzaSyAhh1f5YAVcvBwvnIFL9ZSowBTXLnrE1G0')

# Function to upload file to Gemini
def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Function to wait for files to be active (processed)
def wait_for_files_active(files):
    """Waits for the given files to be active."""
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")

# Streamlit UI for inputs
st.title("Rephrase Customer Queries Using Gemini Model")

# Tabbed interface for rephrasing options
tab1, tab2 = st.tabs(["Query & Response", "Response Only"])

# File uploader for custom dataset
uploaded_file = st.file_uploader("Upload a rephrasing dataset (text file)", type="txt", key="file_uploader")

# Default file path
default_file_path = "miro_data.txt"

# Use uploaded file or default file
if uploaded_file is not None:
    with open("uploaded_data.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    dataset_path = "uploaded_data.txt"
    st.write("Using uploaded file for rephrasing.")
else:
    dataset_path = default_file_path
    st.write("Using default file for rephrasing.")

# Gemini model setup
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="Understand the manner in which we rephrase the response from the file shared and your job is to rephrase the new queries given to you.",
)

# Upload and process the file
files = [upload_to_gemini(dataset_path, mime_type="text/plain")]
wait_for_files_active(files)

# Tab 1: Rephrase using query and response
with tab1:
    st.subheader("Rephrase with Query & Response")
    
    # Input fields for query and response with unique keys
    user_query = st.text_area("Enter the customer query:", key="query_input_tab1")
    user_response = st.text_area("Enter the original response:", key="response_input_tab1")

    # Rephrase the input when button is clicked
    if st.button("Rephrase Response (Query & Response)", key="rephrase_qr_button_tab1"):
        # Start chat session with the uploaded file and user inputs
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                        "Understand the manner in which we rephrase the response from the file shared and your job is to rephrase the new queries given to you.",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "Okay, I understand. You want me to rephrase responses to customer queries based on the pattern you've shown in the `dataset.txt` file. I will not mention names until specified in query or actual response.",
                    ],
                },
            ]
        )

        # Send the new message and get the response
        response = chat_session.send_message(f"query: {user_query}\n\nresponse: {user_response}\n\nJust send back the rephrased response.")

        # Display the rephrased response
        st.subheader("Rephrased Response")
        st.write(response.text)

# Tab 2: Rephrase using only the response
with tab2:
    st.subheader("Rephrase with Response Only")

    # Input field for only the response with a unique key
    user_response_only = st.text_area("Enter the original response:", key="response_only_input_tab2")

    # Rephrase the input when button is clicked
    if st.button("Rephrase Response (Response Only)", key="rephrase_ro_button_tab2"):
        # Start chat session with the uploaded file and user inputs
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                        "Understand the manner in which we rephrase the response from the file shared and your job is to rephrase the new responses given to you.",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "Okay, I understand. You want me to rephrase responses based on the pattern you've shown in the `dataset.txt` file. I will not mention names until specified in the actual response.",
                    ],
                },
            ]
        )

        # Send the new message and get the rephrased response
        response = chat_session.send_message(f"response: {user_response_only}\n\nJust send back the rephrased response.")

        # Display the rephrased response
        st.subheader("Rephrased Response")
        st.write(response.text)
