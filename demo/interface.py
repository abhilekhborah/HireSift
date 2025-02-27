import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv
import io
import uuid

from PIL import Image
import pandas as pd
import streamlit as st
from streamlit_modal import Modal
from pypdf import PdfReader

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity
from interview_agent import InterviewScheduler

load_dotenv()

# Default paths for data storage and models
DEFAULT_DATA_PATH = os.getenv("DATA_PATH", "")
GENERATED_DATA_DIR = os.getenv("GENERATED_DATA_PATH", "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/data/csv")
GENERATED_DATA_PATH = os.path.join(GENERATED_DATA_DIR, "processed_resumes.csv")  # Create a file path, not just a directory
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Create directories if they don't exist
os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

welcome_message = """
  HireSift is a RAG based application developed to assist hiring managers in searching for the most suitable candidates out of a large collection of resumes swiftly. ⚡
  
  You can also schedule interviews directly by typing commands like "schedule an interview for John Doe at 9am tomorrow" in the chat.
"""

st.set_page_config(page_title="HireSift", layout="wide")
col1, col2 = st.columns([4, 1])
with col1:
    st.title("HireSift")
with col2:
    logo_path = "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/demo/pngwing.com (11).png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=150)
    except FileNotFoundError:
        st.error("Logo image not found. Place 'logo.png' in the same directory as your script.")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
  # Try to load generated data if it exists, otherwise fall back to default data
  # if os.path.exists(GENERATED_DATA_PATH):
  st.session_state.df = ""
  # else:
  #   st.session_state.df = pd.read_csv(DEFAULT_DATA_PATH)

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = VoyageEmbeddings(model="voyage-large-2-instruct", voyage_api_key=os.getenv("VOYAGE_API_KEY"))

if "rag_pipeline" not in st.session_state:
  # Try to load existing FAISS index if it exists, otherwise create new one
  if os.path.exists(FAISS_PATH):
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  else:
    # If no index exists, create one from the current dataframe
    vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
    vectordb.save_local(FAISS_PATH)
    
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []

# Initialize interview scheduler
if "interview_scheduler" not in st.session_state:
    st.session_state.interview_scheduler = InterviewScheduler()

# Track email send status
if "email_sent_status" not in st.session_state:
    st.session_state.email_sent_status = {}

def process_pdf_files(pdf_files):
    """Process uploaded PDF files and convert to a DataFrame with ID and Resume columns"""
    data = []
    for i, pdf_file in enumerate(pdf_files):
        pdf_bytes = pdf_file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        data.append({"ID": i, "Resume": text})
    
    # Save the processed data to the configured path - using a file path, not a directory
    df = pd.DataFrame(data)
    df.to_csv(GENERATED_DATA_PATH, index=False)
    
    return df

def upload_file():
    modal = Modal(key="Demo Key", title="File Error", max_width=500)
    
    if st.session_state.uploaded_files:
        try:
            # Process PDF files into a DataFrame
            with st.toast('Processing PDF files...'):
                df_load = process_pdf_files(st.session_state.uploaded_files)
            
            with st.toast('Indexing the uploaded data. This may take a while...'):
                st.session_state.df = df_load
                vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
                # Save the new index
                vectordb.save_local(FAISS_PATH)
                st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
                
                # Show success message
                st.success(f"Successfully processed {len(df_load)} PDF resumes and created vector index.")
                
        except Exception as error:
            with modal.container():
                st.markdown("An error occurred while processing the PDF files:")
                st.error(error)
    else:
        # Revert to default data if no files are uploaded
        if os.path.exists(DEFAULT_DATA_PATH):
            st.session_state.df = pd.read_csv(DEFAULT_DATA_PATH)
            if os.path.exists(FAISS_PATH):
                vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, 
                                          distance_strategy=DistanceStrategy.COSINE, 
                                          allow_dangerous_deserialization=True)
            else:
                vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
                vectordb.save_local(FAISS_PATH)
                
            st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
        else:
            st.error("Default data file not found. Please upload PDF resumes to continue.")

def check_groq_api_key(api_key: str):
    try:
        _ = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=4000,
            streaming=True
        )
        return True
    except Exception:
        return False
  
def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

def send_email_callback(candidate_id):
    """Send an email to the candidate and update status"""
    # Mock sending the email
    send_success = st.session_state.interview_scheduler.send_email(candidate_id)
    if send_success:
        st.session_state.email_sent_status[candidate_id] = True
        st.experimental_rerun()
    else:
        st.session_state.email_sent_status[candidate_id] = False
        st.experimental_rerun()

user_query = st.chat_input("Type your message here...")

# Simplified sidebar with just the essentials
with st.sidebar:
    st.markdown("# Control Panel")
    st.selectbox("RAG Mode", ["RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
    st.text_input("Model", "gemini-2.0-flash", key="llm_selection")
    st.file_uploader("Upload resumes", type=["pdf"], accept_multiple_files=True, key="uploaded_files", on_change=upload_file)
    st.button("Clear conversation", on_click=clear_message)

# Single tab for all functionality - simplified interface
st.header("Resume Search & Interview Scheduling")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
    else:
        with st.chat_message("AI"):
            message[0].render(*message[1:])

# Set up retriever and LLM
retriever = st.session_state.rag_pipeline

llm = ChatBot(
    api_key=os.getenv("GROQ_API_KEY"),
    model=st.session_state.llm_selection,
)


# Process user query
if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
        # Check if this is an interview scheduling request
        interview_request = llm.detect_interview_request(user_query)
        
        bulk_scheduling_request = llm.detect_bulk_scheduling_request(user_query)

        if bulk_scheduling_request.get("is_bulk_scheduling", False):
            # Process bulk interview scheduling
            with st.spinner("Scheduling multiple interviews..."):
                try:
                    # Make sure we have resume data before scheduling
                    if not st.session_state.resume_list:
                        st.warning("No candidate data found. Please search for candidates first.")
                        response = "I need candidate information before I can schedule interviews. Please search for candidates first by asking about specific roles or skills."
                    else:
                        scheduling_result = llm.schedule_bulk_interviews(
                            st.session_state.interview_scheduler,
                            bulk_scheduling_request,
                            st.session_state.resume_list
                        )
                        
                        if scheduling_result["success"]:
                            interviews = scheduling_result["interviews"]
                            invitation_drafts = scheduling_result.get("invitation_drafts", [])
                            
                            # Display confirmation message
                            st.success(f"Successfully scheduled {len(interviews)} interviews!")
                            
                            # Create a styled container for all interview details
                            with st.container():
                                st.markdown("### 📅 Interview Schedule")
                                
                                for i, interview in enumerate(interviews):
                                    # Create an expander for each interview
                                    with st.expander(f"Interview for {interview.candidate_name}", expanded=True):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown(f"**Candidate:** {interview.candidate_name}")
                                            st.markdown(f"**Date:** {interview.date}")
                                        with col2:
                                            st.markdown(f"**Interviewer:** {interview.interviewer_name}")
                                            st.markdown(f"**Time:** {interview.start_time}-{interview.end_time}")
                                        
                                        st.markdown(f"**Meeting Link:** [{interview.meeting_link}]({interview.meeting_link})")
                                        
                                        # Find matching invitation draft
                                        if invitation_drafts:
                                            matching_draft = next(
                                                (draft for draft in invitation_drafts 
                                                if draft["candidate_id"] == interview.candidate_id),
                                                None
                                            )
                                            
                                            if matching_draft:
                                                # Display email draft
                                                st.markdown("#### 📧 Email Draft")
                                                st.text_area(
                                                    "Email Content", 
                                                    matching_draft["email_draft"], 
                                                    height=200, 
                                                    key=f"draft_{interview.candidate_id}"
                                                )
                                                
                                                # Email sending functionality
                                                candidate_id = interview.candidate_id
                                                if candidate_id in st.session_state.email_sent_status:
                                                    if st.session_state.email_sent_status[candidate_id]:
                                                        st.success("✅ Email sent successfully!")
                                                    else:
                                                        st.error("❌ Failed to send email.")
                                                        if st.button("Retry Sending Email", key=f"retry_{candidate_id}"):
                                                            send_email_callback(candidate_id)
                                                else:
                                                    if st.button("Send Email", key=f"send_{candidate_id}"):
                                                        send_email_callback(candidate_id)
                            
                            response = scheduling_result["message"]
                        else:
                            st.error(scheduling_result["message"])
                            response = f"I couldn't schedule the interviews: {scheduling_result['message']}"
                except Exception as e:
                    st.error(f"Error during interview scheduling: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                    response = "There was an error while scheduling the interviews. Please check the logs."
                    
        elif interview_request.get("is_interview_request", False):
            # Process interview scheduling
            with st.spinner("Scheduling interview..."):
                try:
                    # Make sure we have resume data before scheduling
                    if not st.session_state.resume_list:
                        st.warning("No candidate data found. Please search for candidates first.")
                        response = "I need candidate information before I can schedule interviews. Please search for candidates first by asking about specific roles or skills."
                    else:
                        scheduling_result = llm.schedule_interview_for_candidate(
                            st.session_state.interview_scheduler,
                            interview_request,
                            st.session_state.resume_list
                        )
                        
                        if scheduling_result["success"]:
                            interview = scheduling_result["interview"]
                            
                            # Generate email draft
                            email_draft = st.session_state.interview_scheduler.get_email_draft(interview)
                            
                            # Create a unique key for this interview
                            candidate_id = interview.candidate_id
                            
                            # Display confirmation message with meeting details
                            st.success(f"Interview scheduled successfully!")
                            
                            # Create a styled container for the interview details
                            with st.container():
                                st.markdown("### 📅 Interview Details")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Candidate:** {interview.candidate_name}")
                                    st.markdown(f"**Date:** {interview.date}")
                                with col2:
                                    st.markdown(f"**Interviewer:** {interview.interviewer_name}")
                                    st.markdown(f"**Time:** {interview.start_time}-{interview.end_time}")
                                
                                st.markdown(f"**Meeting Link:** [{interview.meeting_link}]({interview.meeting_link})")
                                
                                # Display email draft in a collapsible section
                                with st.expander("📧 Email Draft", expanded=True):
                                    st.text_area("Email Content", email_draft, height=200, key=f"draft_{candidate_id}")
                                    
                                    # Check if email has been sent
                                    if candidate_id in st.session_state.email_sent_status:
                                        if st.session_state.email_sent_status[candidate_id]:
                                            st.success("✅ Email sent successfully!")
                                        else:
                                            st.error("❌ Failed to send email.")
                                            if st.button("Retry Sending Email", key=f"retry_{candidate_id}"):
                                                send_email_callback(candidate_id)
                                    else:
                                        if st.button("Send Email", key=f"send_{candidate_id}"):
                                            send_email_callback(candidate_id)
                            
                            response = scheduling_result["message"]
                        else:
                            st.error(scheduling_result["message"])
                            response = f"I couldn't schedule the interview: {scheduling_result['message']}"
                except Exception as e:
                    st.error(f"Error during interview scheduling: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                    response = "There was an error while scheduling the interview. Please check the logs."
        else:
            # Normal query processing for resume search
            start = time.time()
            with st.spinner("Generating answers..."):
                document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
                query_type = retriever.meta_data["query_type"]
                st.session_state.resume_list = document_list
                stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
            end = time.time()

            response = st.write_stream(stream_message)
            
            retriever_message = chatbot_verbosity
            retriever_message.render(document_list, retriever.meta_data, end-start)

        st.session_state.chat_history.append(AIMessage(content=response))
        
        # Only add retriever_message if it was a normal query, not an interview request
        if not interview_request.get("is_interview_request", False):
            st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))