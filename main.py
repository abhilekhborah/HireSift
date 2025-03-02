import sys, os
sys.dont_write_bytecode = True

import time
import logging
from dotenv import load_dotenv
import io
import uuid
import json
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import pandas as pd
import streamlit as st
from streamlit_modal import Modal
from pypdf import PdfReader

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from llm_agent import ChatBot
from ingest_data import ingest, save_vectorstore, load_vectorstore
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity
from interview_agent import InterviewScheduler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('hiresift_interface')

# Load environment variables
load_dotenv()

# Debug mode - set to False in production
DEBUG_MODE = False

# Default paths for data storage and models
DEFAULT_DATA_PATH = os.getenv("DATA_PATH", "")
GENERATED_DATA_DIR = os.getenv("GENERATED_DATA_PATH", "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/data/csv")
GENERATED_DATA_PATH = os.path.join(GENERATED_DATA_DIR, "processed_resumes.csv")  # Create a file path, not just a directory
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Create directories if they don't exist
os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
if FAISS_PATH:
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

# Improved welcome message with clearer formatting and instructions
welcome_message = """
HireSift helps you quickly find and schedule interviews with the most qualified candidates from your resume database.

## What you can do:
- **Search for candidates**: "Find me 3 fullstack developers with React experience"
- **Schedule interviews**: "Schedule an interview with candidate #1 tomorrow at 2pm"
- **Bulk scheduling**: "Schedule interviews for all the selected candidates on Thursday"

*Upload your resume database using the sidebar to get started.*
"""

# Conversation templates for consistent response formatting
conversation_templates = {
    # Template for candidate search results
    "candidate_search_results": """
# Candidates Matching Your Search 🔍

I've found {count} candidates that match your requirements for the {role_type} position:

{candidate_summaries}

Would you like to schedule interviews with any of these candidates?
""",
    
    # Template for when no candidates are found
    "no_candidates_found": """
I couldn't find any candidates matching your specific requirements in our database.

### Suggestions:
- Try broadening your search with fewer specific skills
- Search for a related role type
- Use more common skill keywords
- Upload more resumes to increase the candidate pool

Would you like to try a different search?
""",
    
    # Template for single interview confirmation
    "interview_scheduled": """
# Interview Scheduled ✅

Successfully scheduled an interview for **{candidate_name}** with **{interviewer_name}**.

### Interview Details:
- **Date:** {formatted_date}
- **Time:** {formatted_time}
- **Format:** Video Conference
- **Meeting Link:** [Join Meeting]({meeting_link})

""",
    
    # Template for bulk interview confirmation
"bulk_interviews_scheduled": """
# Multiple Interviews Scheduled ✅

Successfully scheduled interviews for **{count}** candidates.

All candidates have been assigned interviewers based on skill matching and interviewer availability.
""",

    
    # Template for interview error
    "interview_error": """
I encountered an issue while trying to schedule the interview:

**Error:** {error_message}

Please try again with different parameters or contact support if the issue persists.
""",


# Add these new templates to the conversation_templates dictionary in interface.py

    # Template for category-based search results
    "category_search_results": """
# Candidates Matching Your Categories 🔍

Here are the candidates matching your requested categories:

{category_summaries}

Would you like to schedule interviews with any of these candidates?
""",

    # Template for category-based partial results
    "partial_category_results": """
# Candidates Matching Your Categories 🔍

I've searched for candidates matching your requested categories:

{category_summaries}

{note}

Would you like to schedule interviews with any of these available candidates?
""",

    # Template for partial results when fewer candidates found than requested
    "partial_results": """
# Candidates Matching Your Search 🔍

I found {found_count} candidate{found_plural} that match{singular_verb} your requirements for the {role_type} position (you requested {requested_count}):

{candidate_summaries}

Would you like to schedule {found_plural_verb} with {found_plural_pronoun}?
"""

}

# Setup the Streamlit page
st.set_page_config(page_title="HireSift", layout="wide")
col1, col2 = st.columns([4, 1])
with col1:
    st.title("HireSift")
with col2:
    logo_path = "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/siemens_logo.png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=150)
    except FileNotFoundError:
        st.error("Logo image not found. Place 'logo.png' in the same directory as your script.")

# Initialize session state if needed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
    # Try to load generated data if it exists, otherwise create an empty DataFrame
    if os.path.exists(GENERATED_DATA_PATH):
        try:
            st.session_state.df = pd.read_csv(GENERATED_DATA_PATH)
            logger.info(f"Loaded data from {GENERATED_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.session_state.df = pd.DataFrame(columns=["ID", "Resume"])
    else:
        st.session_state.df = pd.DataFrame(columns=["ID", "Resume"])

if "embedding_model" not in st.session_state:
    try:
        st.session_state.embedding_model = VoyageEmbeddings(model="voyage-large-2-instruct", voyage_api_key=os.getenv("VOYAGE_API_KEY"))
        logger.info("Initialized VoyageEmbeddings model")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        # Fallback to a simpler model
        try:
            st.session_state.embedding_model = HuggingFaceEmbeddings()
            logger.info("Initialized fallback HuggingFaceEmbeddings model")
        except Exception as inner_e:
            logger.error(f"Error initializing fallback embedding model: {inner_e}")
            st.error("Failed to initialize embedding model. Some features may not work correctly.")

if "rag_pipeline" not in st.session_state:
    # Try to load existing FAISS index if it exists, otherwise create new one
    if os.path.exists(FAISS_PATH):
        try:
            vectordb = load_vectorstore(FAISS_PATH, st.session_state.embedding_model)
            if vectordb is None:
                # If loading fails, create a new vectorstore
                vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
                save_vectorstore(vectordb, FAISS_PATH)
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
            save_vectorstore(vectordb, FAISS_PATH)
    else:
        # If no index exists, create one from the current dataframe
        vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
        save_vectorstore(vectordb, FAISS_PATH)
        
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []

# Initialize interview scheduler
if "interview_scheduler" not in st.session_state:
    st.session_state.interview_scheduler = InterviewScheduler()

# Track email send status
if "email_sent_status" not in st.session_state:
    st.session_state.email_sent_status = {}

# Initialize evaluation metrics
if "evaluation_metrics" not in st.session_state:
    st.session_state.evaluation_metrics = {
        "shortlisting_accuracy": [],
        "contact_extraction_efficiency": [],
        "invitation_correctness": [],
        "scheduling_fairness": []
    }

def process_query_results(retriever, docs, query, required_count=None):
    """
    Process and format query results
    
    Args:
        retriever: The retriever instance with metadata
        docs: Retrieved documents
        query: Original user query
        required_count: Number of candidates requested
        
    Returns:
        Formatted response using the appropriate template
    """
    if not docs:
        # No candidates found
        role_info = extract_role_and_skills_from_query(query)
        role_type = role_info.get("role_type", "candidates")
        return conversation_templates["no_candidates_found"].format(role_type=role_type)
    
    # Preserve shortlisting reasons between queries
    if hasattr(retriever, "meta_data") and "shortlisting_reasons" in retriever.meta_data:
        # Initialize storage for accumulated shortlisting reasons if not exists
        if "all_shortlisting_reasons" not in st.session_state:
            st.session_state.all_shortlisting_reasons = {}
        
        # Add new shortlisting reasons to accumulated set
        for doc_id, reason_data in retriever.meta_data["shortlisting_reasons"].items():
            st.session_state.all_shortlisting_reasons[doc_id] = reason_data
        
        # Update retriever metadata to include all accumulated reasons
        retriever.meta_data["shortlisting_reasons"] = st.session_state.all_shortlisting_reasons
    
    # Check if this is a category-based query
    categories = retriever.meta_data.get("categories", {})
    if categories:
        # Extract categories from retrieved documents
        found_categories = {}
        for doc in docs:
            category_match = re.search(r'Category:\s+(\w+)', doc)
            if category_match:
                category = category_match.group(1)
                if category not in found_categories:
                    found_categories[category] = []
                found_categories[category].append(doc)
        
        # Create category summaries
        category_summaries = ""
        missing_categories = []
        
        for category, count in categories.items():
            found = found_categories.get(category, [])
            found_count = len(found)
            
            if found_count > 0:
                category_summaries += f"\n## {category.capitalize()} Candidates ({found_count}/{count} found)\n\n"
                for doc in found:
                    # Extract candidate information for summary
                    id_match = re.search(r'Applicant ID\s+(\d+)', doc)
                    candidate_id = id_match.group(1) if id_match else "Unknown"
                    
                    # Create a summary for this candidate
                    category_summaries += f"### Candidate #{candidate_id}\n"
                    
                    # Extract skills if available
                    skills_match = re.search(r'Skills:\s+([^\n]+)', doc)
                    if skills_match:
                        skills = skills_match.group(1)
                        category_summaries += f"**Skills:** {skills}\n\n"
                    
                    # Add a summary of the candidate (first 150 chars)
                    resume_text = doc.split("\n", 4)[-1] if len(doc.split("\n")) > 4 else ""
                    summary = resume_text[:150] + "..." if len(resume_text) > 150 else resume_text
                    category_summaries += f"{summary}\n\n"
            else:
                missing_categories.append(category)
                
        # Check if we're missing any categories
        if missing_categories:
            note = f"**Note:** I couldn't find any candidates for the following categories: {', '.join(missing_categories)}."
            template = conversation_templates["partial_category_results"].format(
                category_summaries=category_summaries,
                note=note
            )
        else:
            template = conversation_templates["category_search_results"].format(
                category_summaries=category_summaries
            )
            
        return template
    
    # Handle standard query (non-category)
    if required_count is None:
        required_count = retriever.meta_data.get("required_count", 1)
        
    found_count = len(docs)
    
    # Create candidate summaries
    candidate_summaries = ""
    for doc in docs:
        # Extract candidate ID
        id_match = re.search(r'Applicant ID\s+(\d+)', doc)
        candidate_id = id_match.group(1) if id_match else "Unknown"
        
        # Create a summary
        candidate_summaries += f"### Candidate #{candidate_id}\n"
        
        # Extract skills
        skills_match = re.search(r'Skills:\s+([^\n]+)', doc)
        if skills_match:
            skills = skills_match.group(1)
            candidate_summaries += f"**Skills:** {skills}\n\n"
        
        # Add a summary
        resume_text = doc.split("\n", 4)[-1] if len(doc.split("\n")) > 4 else ""
        summary = resume_text[:150] + "..." if len(resume_text) > 150 else resume_text
        candidate_summaries += f"{summary}\n\n"
    
    # Determine role type
    role_info = extract_role_and_skills_from_query(query)
    role_type = role_info.get("role_type", "candidate")
    
    # Check if we found enough candidates
    if found_count < required_count:
        # Partial results
        singular_verb = "" if found_count > 1 else "es"
        plural = "s" if found_count > 1 else ""
        plural_verb = "interviews" if found_count > 1 else "an interview"
        plural_pronoun = "them" if found_count > 1 else "this candidate"
        
        template = conversation_templates["partial_results"].format(
            found_count=found_count,
            requested_count=required_count,
            role_type=role_type,
            candidate_summaries=candidate_summaries,
            found_plural=plural,
            singular_verb=singular_verb,
            found_plural_verb=plural_verb,
            found_plural_pronoun=plural_pronoun
        )
    else:
        # Full results
        singular_verb = "" if found_count > 1 else "es"
        plural = "s" if found_count > 1 else ""
        
        template = conversation_templates["candidate_search_results"].format(
            count=found_count,
            role_type=role_type,
            candidate_summaries=candidate_summaries,
            plural=plural,
            singular_verb=singular_verb
        )
        
    return template

def process_pdf_files(pdf_files):
    """Process uploaded PDF files and convert to a DataFrame with ID and Resume columns"""
    data = []
    try:
        for i, pdf_file in enumerate(pdf_files):
            try:
                pdf_bytes = pdf_file.read()
                pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                data.append({"ID": i, "Resume": text})
            except Exception as e:
                logger.error(f"Error processing PDF file {pdf_file.name}: {e}")
                st.error(f"Error processing PDF file {pdf_file.name}: {str(e)}")
        
        # Save the processed data to the configured path - using a file path, not a directory
        if data:
            df = pd.DataFrame(data)
            df.to_csv(GENERATED_DATA_PATH, index=False)
            return df
        else:
            return pd.DataFrame(columns=["ID", "Resume"])
    except Exception as e:
        logger.error(f"Error in process_pdf_files: {e}")
        st.error(f"Error processing PDF files: {str(e)}")
        return pd.DataFrame(columns=["ID", "Resume"])

def extract_resume_features(resume_df):
    """Extract features from resumes for better shortlisting and evaluation"""
    # Create copy to avoid modifying original
    enhanced_df = resume_df.copy()
    
    # Initialize new columns
    enhanced_df["Skills"] = None
    enhanced_df["Years_Experience"] = None
    enhanced_df["Education"] = None
    enhanced_df["Email"] = None
    enhanced_df["Phone"] = None
    enhanced_df["Name"] = None
    
    # Process each resume
    for idx, row in enhanced_df.iterrows():
        resume_text = row["Resume"]
        
        # Extract skills
        skills = extract_skills_from_text(resume_text)
        enhanced_df.at[idx, "Skills"] = json.dumps(skills)  # Store as JSON for DataFrame compatibility
        
        # Extract years of experience
        years_exp = extract_experience_years(resume_text)
        enhanced_df.at[idx, "Years_Experience"] = years_exp
        
        # Extract education
        education = extract_education_level(resume_text)
        enhanced_df.at[idx, "Education"] = education
        
        # Extract contact info
        email = extract_email(resume_text)
        enhanced_df.at[idx, "Email"] = email
        
        phone = extract_phone(resume_text)
        enhanced_df.at[idx, "Phone"] = phone
        
        name = extract_name(resume_text)
        enhanced_df.at[idx, "Name"] = name
    
    return enhanced_df

def extract_skills_from_text(text):
    """Extract skills from resume text"""
    skill_keywords = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Ruby", "PHP",
        "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring", "Express",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins",
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "NoSQL", "Redis", "GraphQL",
        "Machine Learning", "Deep Learning", "AI", "NLP", "Computer Vision", "GenAI",
        "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "BERT", "GPT",
        "HTML", "CSS", "SASS", "LESS", "Bootstrap", "Tailwind",
        "Git", "GitHub", "GitLab", "CI/CD", "Agile", "Scrum", "Jira",
        "Fullstack", "Frontend", "Backend", "DevOps", "Mobile", "Android", "iOS"
    ]
    
    found_skills = []
    for skill in skill_keywords:
        # Look for whole word match with word boundaries
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)
    
    return found_skills

def extract_experience_years(text):
    """Estimate years of experience from resume text"""
    # Look for patterns like "X years of experience"
    exp_patterns = [
        r'(\d+)\+?\s*(?:years|yrs)(?:\s+of)?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s*(?:years|yrs)',
        r'(?:over|more\s+than)\s+(\d+)\s*(?:years|yrs)'
    ]
    
    for pattern in exp_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                years = int(match.group(1))
                if 0 < years < 50:  # Sanity check
                    return years
            except ValueError:
                continue
    
    # Look for work history dates
    date_pattern = r'(\d{4})\s*[-–—to]+\s*(?:(\d{4})|present|current)'
    date_matches = re.findall(date_pattern, text, re.IGNORECASE)
    
    if date_matches:
        # Use the current year for "present" or "current"
        current_year = 2025  # Using current year from prompt
        total_years = 0
        
        for match in date_matches:
            start_year = int(match[0])
            if match[1]:  # End year is specified
                end_year = int(match[1])
            else:  # "present" or "current"
                end_year = current_year
                
            # Only count reasonable date ranges
            if 1980 <= start_year <= current_year and start_year <= end_year:
                total_years += min(end_year - start_year, 10)  # Cap single job at 10 years
        
        if total_years > 0:
            return min(total_years, 40)  # Cap total at 40 years
    
    # Default if no patterns match
    return 2

def extract_education_level(text):
    """Extract education level from resume text"""
    education_patterns = {
        "PhD": [r'ph\.?d\.?', r'doctor of philosophy', r'doctorate'],
        "Master's": [r'master', r'm\.?s\.?', r'm\.?a\.?', r'mba', r'm\.?b\.?a\.?'],
        "Bachelor's": [r'bachelor', r'b\.?s\.?', r'b\.?a\.?', r'b\.?tech', r'undergraduate'],
        "Associate's": [r'associate', r'a\.?s\.?', r'a\.?a\.?'],
        "High School": [r'high school', r'secondary school', r'h\.?s\.?']
    }
    
    for level, patterns in education_patterns.items():
        for pattern in patterns:
            if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE):
                return level
    
    # Default if no patterns match
    return "Bachelor's"

def extract_email(text):
    """Extract email address from resume text"""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    if match:
        return match.group(0)
    return None

def extract_phone(text):
    """Extract phone number from resume text"""
    phone_patterns = [
        r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'(?:\+\d{1,3}[-.\s]?)?\d{5}[-.\s]?\d{5}'
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

def extract_name(text):
    """Extract candidate name from resume text"""
    # Try to find name at the beginning of the resume
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    if non_empty_lines:
        # Check if the first line looks like a name (typically the case in resumes)
        name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$'
        match = re.match(name_pattern, non_empty_lines[0])
        if match:
            return match.group(1)
    
    # Try to find explicit name fields
    name_patterns = [
        r'Name[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'Full Name[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def upload_file():
    """Process uploaded files and update the vectorstore"""
    modal = Modal(key="Demo Key", title="File Error", max_width=500)
    
    if st.session_state.uploaded_files:
        try:
            # Process PDF files into a DataFrame
            with st.toast('Processing PDF files...'):
                df_load = process_pdf_files(st.session_state.uploaded_files)
            
            if not df_load.empty:
                with st.toast('Extracting resume features...'):
                    # Extract features from resumes
                    enhanced_df = extract_resume_features(df_load)
                
                with st.toast('Indexing the uploaded data. This may take a while...'):
                    st.session_state.df = enhanced_df
                    vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
                    # Save the new index
                    save_vectorstore(vectordb, FAISS_PATH)
                    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
                    
                    # Show success message
                    st.success(f"Successfully processed {len(enhanced_df)} PDF resumes and created vector index.")
                    
                    # Display extracted features summary
                    st.info("Extracted resume features summary:")
                    feature_summary = {
                        "Skills Extracted": sum(1 for skills in enhanced_df["Skills"] if skills is not None and skills != "[]"),
                        "Contact Info Extracted": sum(1 for email in enhanced_df["Email"] if email is not None),
                        "Education Extracted": sum(1 for edu in enhanced_df["Education"] if edu is not None),
                        "Experience Extracted": sum(1 for exp in enhanced_df["Years_Experience"] if exp is not None),
                    }
                    
                    # Calculate contact extraction efficiency for evaluation metrics
                    if len(enhanced_df) > 0:
                        contact_efficiency = (feature_summary["Contact Info Extracted"] / len(enhanced_df)) * 100
                        st.session_state.evaluation_metrics["contact_extraction_efficiency"].append(contact_efficiency)
                    
                    for feature, count in feature_summary.items():
                        percentage = (count / len(enhanced_df)) * 100 if len(enhanced_df) > 0 else 0
                        st.write(f"- {feature}: {count}/{len(enhanced_df)} ({percentage:.1f}%)")
            else:
                st.warning("No data was extracted from the uploaded files.")
                
        except Exception as error:
            logger.error(f"Error in upload_file: {error}")
            with modal.container():
                st.markdown("An error occurred while processing the PDF files:")
                st.error(error)
    else:
        # Revert to default data if no files are uploaded
        if os.path.exists(DEFAULT_DATA_PATH):
            try:
                st.session_state.df = pd.read_csv(DEFAULT_DATA_PATH)
                if os.path.exists(FAISS_PATH):
                    vectordb = load_vectorstore(FAISS_PATH, st.session_state.embedding_model)
                    if vectordb is None:
                        vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
                        save_vectorstore(vectordb, FAISS_PATH)
                else:
                    vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
                    save_vectorstore(vectordb, FAISS_PATH)
                    
                st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
            except Exception as e:
                logger.error(f"Error loading default data: {e}")
                st.error("Error loading default data. Please upload PDF resumes to continue.")
        else:
            st.warning("Default data file not found. Please upload PDF resumes to continue.")

def on_new_query():
    """Reset the verbosity rendered flag when a new query is submitted"""
    if hasattr(st.session_state, 'verbosity_rendered'):
        st.session_state.verbosity_rendered = False

def check_groq_api_key(api_key: str):
    """Verify that the Groq API key is valid"""
    try:
        _ = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=4000,
            streaming=True
        )
        return True
    except Exception as e:
        logger.error(f"Error validating Groq API key: {e}")
        return False
  
def clear_message():
    """Clear the chat history and resume list"""
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content=welcome_message)]
    logger.info("Chat history and resume list cleared")

def send_email_callback(candidate_id):
    """Send an email to the candidate and update status"""
    # Mock sending the email
    send_success = st.session_state.interview_scheduler.send_email(candidate_id)
    if send_success:
        st.session_state.email_sent_status[candidate_id] = True
        logger.info(f"Email sent successfully to candidate {candidate_id}")
        st.experimental_rerun()
    else:
        st.session_state.email_sent_status[candidate_id] = False
        logger.warning(f"Failed to send email to candidate {candidate_id}")
        st.experimental_rerun()

# Replace the extract_required_count_from_query function with this improved version

def extract_required_count_from_query(query):
    """
    Extract the number of candidates requested with improved accuracy
    
    Args:
        query: The user query string
        
    Returns:
        Number of candidates requested (defaults to 1 if not specified)
    """
    query_lower = query.lower()
    
    # Check for explicit "one" patterns
    one_patterns = ["only one", "just one", "a single", "one good fit", "1 candidate", "one candidate", "a candidate"]
    if any(pattern in query_lower for pattern in one_patterns):
        return 1
            
    # Check for explicit numeric indicators
    count_patterns = [
        r'(\d+)\s+candidates',
        r'(\d+)\s+developers',
        r'(\d+)\s+engineers',
        r'(\d+)\s+professionals',
        r'(\d+)\s+people',
        r'find\s+(\d+)',
        r'get\s+(\d+)',
        r'show\s+(\d+)',
        r'provide\s+(\d+)',
        r'top\s+(\d+)',
        r'(\d+)\s+profiles',
        r'(\d+)\s+resumes'
    ]
    
    for pattern in count_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                count = int(match.group(1))
                # Sanity check - keep count within reasonable bounds
                if 1 <= count <= 10:  # Reasonable upper limit for number of candidates
                    return count
            except:
                pass
    
    # Check for textual number indicators
    text_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    
    for text, number in text_numbers.items():
        patterns = [
            f"{text} candidates",
            f"{text} profiles",
            f"{text} resumes",
            f"{text} people",
            f"find {text}",
            f"get {text}",
            f"show {text}",
            f"{text} developers",
            f"{text} engineers"
        ]
        
        for pattern in patterns:
            if pattern in query_lower:
                return number
    
    # Check if this might be a category-based query
    if re.search(r'\d+\s+(?:for|of)\s+[\w\s-]+', query_lower) or re.search(r'[\w\s-]+:\s*\d+', query_lower):
        return None
    
    # If query contains "a" or "an" followed by role, assume 1
    if re.search(r'\ba\s+|\ban\s+', query_lower):
        return 1
        
    # Check for "and" pattern indicating multiple categories
    if "and" in query_lower and any(role in query_lower for role in ["genai", "gen ai", "fullstack", "full stack", "frontend", "backend"]):
        return None
    
    # If no count specified, return 1 for focused results
    return 1


# Replace the extract_categories_from_query function with this improved version

def extract_categories_from_query(query):
    """
    Extract multiple job categories and counts from the query with enhanced pattern matching
    
    Args:
        query: The user query string
        
    Returns:
        Dictionary mapping category names to counts
    """
    categories = {}
    query_lower = query.lower()
    
    # Pattern for "X for Y" (e.g., "2 for GenAI")
    pattern1 = r'(\d+)\s+(?:for|of)\s+([\w\s-]+?)(?:[,\.]|$|\s+and\s+)'
    # Pattern for "Y: X" (e.g., "GenAI: 2")
    pattern2 = r'([\w\s-]+?):\s*(\d+)(?:[,\.]|$|\s+and\s+)'
    # Pattern for "X Y" (e.g., "2 GenAI")
    pattern3 = r'(\d+)\s+([\w\s-]+?)(?:\s+(?:developer|candidate|engineer|specialist|professional)s?)?(?:[,\.]|$|\s+and\s+)'
    
    # First look for explicit category specifications using all patterns
    all_matches = []
    for pattern in [pattern1, pattern2, pattern3]:
        matches = re.finditer(pattern, query_lower)
        for match in matches:
            if pattern == pattern2:  # "Y: X" pattern
                category = match.group(1).strip()
                count = match.group(2)
            else:  # "X for Y" or "X Y" patterns
                count = match.group(1)
                category = match.group(2).strip()
            
            try:
                count = int(count)
                # Normalize category name
                category = re.sub(r'[-_]', ' ', category).strip()
                # Store if reasonable count
                if 1 <= count <= 10 and category:
                    all_matches.append((category, count))
            except ValueError:
                continue
    
    # Process matches to combine similar categories
    for category, count in all_matches:
        # Normalize common categories
        if re.match(r'gen\s*ai|generative\s*ai', category):
            norm_category = "genai"
        elif re.match(r'full\s*stack|full-stack', category):
            norm_category = "fullstack"
        elif re.match(r'front\s*end', category):
            norm_category = "frontend"
        elif re.match(r'back\s*end', category):
            norm_category = "backend"
        else:
            norm_category = category
            
        # Add or update category
        if norm_category in categories:
            categories[norm_category] += count
        else:
            categories[norm_category] = count
    
    # Check for "and" pattern indicating multiple categories
    # Look for patterns like "a fullstack developer and 2 genai engineers"
    if "and" in query_lower and not all_matches:
        parts = re.split(r'\s+and\s+', query_lower)
        for i, part in enumerate(parts):
            # Check first part for singular
            if i == 0 and re.search(r'\ba\s+|\ban\s+', part):
                # Find what role it's referring to
                role_match = re.search(r'\b(?:a|an)\s+([\w\s-]+)(?:\s+(?:developer|engineer|candidate|specialist))?', part)
                if role_match:
                    role = role_match.group(1).strip()
                    # Normalize the role
                    if re.match(r'gen\s*ai|generative\s*ai', role):
                        norm_role = "genai"
                    elif re.match(r'full\s*stack|full-stack', role):
                        norm_role = "fullstack"
                    elif re.match(r'front\s*end', role):
                        norm_role = "frontend"
                    elif re.match(r'back\s*end', role):
                        norm_role = "backend"
                    else:
                        norm_role = role
                        
                    # Add with count 1
                    if norm_role not in categories:
                        categories[norm_role] = 1
            
            # Check for count in other parts
            count_match = re.search(r'(\d+)\s+([\w\s-]+)', part)
            if count_match:
                try:
                    count = int(count_match.group(1))
                    role = count_match.group(2).strip()
                    
                    # Normalize role
                    if re.match(r'gen\s*ai|generative\s*ai', role):
                        norm_role = "genai"
                    elif re.match(r'full\s*stack|full-stack', role):
                        norm_role = "fullstack"
                    elif re.match(r'front\s*end', role):
                        norm_role = "frontend"
                    elif re.match(r'back\s*end', role):
                        norm_role = "backend"
                    else:
                        norm_role = role
                        
                    # Add to categories
                    if norm_role not in categories:
                        categories[norm_role] = count
                except ValueError:
                    continue
    
    # If no explicit categories found, try to infer from the query
    if not categories:
        # Check for commonly requested roles
        common_roles = {
            "genai": ["genai", "gen ai", "generative ai", "llm", "language model"],
            "fullstack": ["fullstack", "full stack", "full-stack"],
            "frontend": ["frontend", "front end", "front-end", "ui", "user interface"],
            "backend": ["backend", "back end", "back-end", "server", "database"],
            "data scientist": ["data scientist", "data science", "analytics"],
            "ai engineer": ["ai engineer", "ai developer", "artificial intelligence"],
            "machine learning": ["machine learning", "ml engineer", "ml developer"]
        }
        
        for role, keywords in common_roles.items():
            if any(keyword in query_lower for keyword in keywords):
                # Look for a number near the keyword
                for keyword in keywords:
                    if keyword in query_lower:
                        # Look for a number before the keyword
                        num_match = re.search(r'(\d+)\s+(?:\w+\s+){0,3}' + re.escape(keyword), query_lower)
                        if num_match:
                            categories[role] = int(num_match.group(1))
                            break
                        else:
                            # If keyword is found but no count, assume 1
                            categories[role] = 1
                            break
    
    return categories

def extract_role_and_skills_from_query(query, llm=None):
    """Extract role type and required skills from a user query"""
    query_lower = query.lower()
    
    # Try LLM-based extraction first if available
    if llm:
        try:
            system_message = SystemMessage(content="""
                You are an expert in understanding job search queries.
                Extract the specific role type and any mentioned skills from the query.
                Return your analysis as a JSON object with keys 'role_type' and 'skills'.
            """)
            
            user_message = HumanMessage(content=f"""
                Analyze this recruitment query: "{query}"
                
                Extract:
                - role_type: the specific job role being requested (e.g. "Full Stack Developer", "Data Scientist")
                - skills: list of any specific skills mentioned as requirements
                
                Return only the JSON object.
            """)
            
            response = llm.llm.invoke([system_message, user_message])
            result = json.loads(response.content)
            return result
        except Exception as e:
            logger.error(f"Error in LLM-based extraction: {e}")
            # Fall back to pattern matching if LLM fails
            pass
    
    # Pattern matching approach as fallback
    result = {
        "role_type": "",
        "skills": []
    }
    
    # Common role patterns
    role_patterns = [
        r"for (?:a|an) ([\w\s]+?) (?:role|position|job)",
        r"([\w\s]+?) developer",
        r"([\w\s]+?) engineer",
        r"([\w\s]+?) specialist",
        r"([\w\s]+?) analyst",
        r"([\w\s]+?) manager",
        r"([\w\s]+?) designer"
    ]
    
    # Try to extract role type
    for pattern in role_patterns:
        match = re.search(pattern, query_lower)
        if match:
            result["role_type"] = match.group(1).strip()
            break
    
    # Common technical skills to extract
    common_skills = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby",
        "react", "angular", "vue", "node", "django", "flask", "spring",
        "aws", "azure", "gcp", "cloud", "docker", "kubernetes",
        "sql", "mysql", "postgresql", "mongodb", "nosql",
        "machine learning", "ai", "data science",
        "frontend", "backend", "fullstack", "full stack", "mobile", "ios", "android",
        "devops", "ci/cd", "jenkins", "agile", "scrum", "genai"
    ]
    
    # Extract mentioned skills
    for skill in common_skills:
        if skill in query_lower:
            result["skills"].append(skill)
    
    return result
def parse_candidate_from_string(candidate_str):
    """
    Improved candidate parsing with better name extraction and fallbacks
    
    Args:
        candidate_str: String representation of a candidate resume
        
    Returns:
        Dictionary with parsed candidate information
    """
    if isinstance(candidate_str, dict):
        # If already a dictionary, ensure it has default values for missing fields
        defaults = {
            "ID": str(random.randint(1000, 9999)),
            "Name": "Candidate",
            "Skills": ["General"]
        }
        
        for key, default_value in defaults.items():
            if key not in candidate_str or not candidate_str[key]:
                candidate_str[key] = default_value
                
        return candidate_str
        
    candidate_data = {}
    
    # Extract ID with multiple patterns
    id_patterns = [
        r'Applicant ID[:\s]*(\d+)',
        r'ID[:\s]*(\d+)',
        r'ID[:\s]*#?(\d+)',
        r'#(\d+)'
    ]
    
    for pattern in id_patterns:
        id_match = re.search(pattern, candidate_str, re.IGNORECASE)
        if id_match:
            candidate_data["ID"] = id_match.group(1)
            break
    
    # If no ID found, generate a random one
    if "ID" not in candidate_data:
        candidate_data["ID"] = str(random.randint(1000, 9999))
    
    # NEW: Check for specific name patterns from candidate sections
    candidate_section_pattern = r'Candidate \d+: ([^(]+)(?:\s*\(Applicant ID (\d+)\))?'
    candidate_section_match = re.search(candidate_section_pattern, candidate_str)
    if candidate_section_match:
        candidate_data["Name"] = candidate_section_match.group(1).strip()
        # If we also got an ID from this pattern and don't have one yet, use it
        if candidate_section_match.group(2) and "ID" not in candidate_data:
            candidate_data["ID"] = candidate_section_match.group(2)
            
    # If no name found yet, try explicit name fields
    if "Name" not in candidate_data:
        explicit_name_patterns = [
            r'Name[:\s]*([^\n]+)',
            r'Full Name[:\s]*([^\n]+)',
            r'Candidate Name[:\s]*([^\n]+)',
            r'Applicant Name[:\s]*([^\n]+)'
        ]
        
        for pattern in explicit_name_patterns:
            name_match = re.search(pattern, candidate_str, re.IGNORECASE)
            if name_match:
                candidate_data["Name"] = name_match.group(1).strip()
                break
    
    # NEW: Look for name with colon followed by text (common in candidate descriptions)
    if "Name" not in candidate_data:
        colon_name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+):'
        colon_name_match = re.search(colon_name_pattern, candidate_str)
        if colon_name_match:
            candidate_data["Name"] = colon_name_match.group(1).strip()
    
    # If still no name, look at the first few lines for a name pattern
    if "Name" not in candidate_data:
        lines = candidate_str.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Check the first few non-empty lines for a name-like pattern
        name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$'  # Pattern for proper names
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Email pattern
        
        for i, line in enumerate(non_empty_lines[:5]):  # Check first 5 non-empty lines
            # Skip lines that look like email addresses
            if re.search(email_pattern, line):
                continue
                
            # Look for name pattern
            name_match = re.search(name_pattern, line)
            if name_match:
                candidate_data["Name"] = name_match.group(1).strip()
                break
    
    # If still no name, use a generic name
    if "Name" not in candidate_data or not candidate_data["Name"]:
        candidate_data["Name"] = f"Candidate {candidate_data['ID']}"
    
    # Extract skills
    skills_found = False
    
    # Try explicit skills section first
    skills_patterns = [
        r'Skills[:\s]*([^\n.]+)',
        r'Technical Skills[:\s]*([^\n.]+)',
        r'Core Competencies[:\s]*([^\n.]+)',
        r'Technologies[:\s]*([^\n.]+)',
        r'Programming Languages[:\s]*([^\n.]+)'
    ]
    
    for pattern in skills_patterns:
        skills_match = re.search(pattern, candidate_str, re.IGNORECASE)
        if skills_match:
            skills_text = skills_match.group(1).strip()
            if skills_text.lower() == "unknown":
                continue
                
            # Parse skills - handle comma-separated or bullet-point lists
            if "," in skills_text:
                candidate_data["Skills"] = [skill.strip() for skill in skills_text.split(',')]
            else:
                # If no commas, try to extract individual skills
                candidate_data["Skills"] = [skills_text]
                
            skills_found = True
            break
    
    # Extract skills from relevant section if present
    if not skills_found:
        skills_section_pattern = r'Relevant Technical Skills and Experience:(.*?)(?:Matching Specific Requirements:|Standout Qualifications:|$)'
        skills_section_match = re.search(skills_section_pattern, candidate_str, re.DOTALL)
        if skills_section_match:
            skills_text = skills_section_match.group(1).strip()
            
            # Extract technical skills using keyword patterns
            common_skills = [
                "Python", "Java", "JavaScript", "React", "Angular", "Vue", "Node.js",
                "HTML", "CSS", "SQL", "NoSQL", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
                "C++", "C#", "Ruby", "PHP", "Go", "Rust", "Swift", "MongoDB", "PostgreSQL",
                "MySQL", "DevOps", "CI/CD", "Git", "Machine Learning", "AI", "GenAI",
                "TensorFlow", "PyTorch", "Agile", "Scrum", "Full Stack", "Frontend", "Backend"
            ]
            
            found_skills = []
            for skill in common_skills:
                # Look for whole word match
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, skills_text, re.IGNORECASE):
                    found_skills.append(skill)
                    
            if found_skills:
                candidate_data["Skills"] = found_skills
                skills_found = True
    
    # If no skills found yet, look for technical keywords throughout the text
    if not skills_found:
        common_skills = [
            "Python", "Java", "JavaScript", "React", "Angular", "Vue", "Node.js",
            "HTML", "CSS", "SQL", "NoSQL", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
            "C++", "C#", "Ruby", "PHP", "Go", "Rust", "Swift", "MongoDB", "PostgreSQL",
            "MySQL", "DevOps", "CI/CD", "Git", "Machine Learning", "AI", "GenAI",
            "TensorFlow", "PyTorch", "Agile", "Scrum", "Full Stack", "Frontend", "Backend"
        ]
        
        found_skills = []
        for skill in common_skills:
            # Look for whole word match
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, candidate_str, re.IGNORECASE):
                found_skills.append(skill)
                
        if found_skills:
            candidate_data["Skills"] = found_skills
            skills_found = True
    
    # If still no skills found, use a generic skill
    if not skills_found or "Skills" not in candidate_data:
        # Check if "Full Stack" is mentioned in the text
        if re.search(r'\bfull\s*stack\b', candidate_str, re.IGNORECASE):
            candidate_data["Skills"] = ["Full Stack"]
        else:
            candidate_data["Skills"] = ["General"]
    
    # Store the original resume text for reference
    candidate_data["Resume"] = candidate_str
    
    return candidate_data

def limit_candidates_to_requested(resume_list, required_count=3):
    """
    Ensure that only the requested number of candidates are included in scheduling
    
    Args:
        resume_list: List of candidate resumes or dictionaries
        required_count: Number of candidates actually requested
        
    Returns:
        Limited list of candidates
    """
    if not resume_list:
        return []
        
    # If we already have the right number, return as is
    if len(resume_list) <= required_count:
        return resume_list
        
    # Otherwise, limit to the required count
    return resume_list[:required_count]

def determine_candidates_to_schedule(resume_list, scheduling_query, original_required_count=3):
    """
    Intelligently determine which candidates to schedule based on the query context
    
    Args:
        resume_list: List of all available candidate resumes
        scheduling_query: The user's scheduling request query
        original_required_count: The number of candidates originally requested in the search
        
    Returns:
        List of candidates to schedule
    """
    if not resume_list:
        return []
        
    # Check if the query explicitly mentions "all" or "each" of the candidates
    schedule_all_keywords = ["all", "each", "every", "everyone", "everybody", "them"]
    mentions_all = any(keyword in scheduling_query.lower() for keyword in schedule_all_keywords)
    
    # Check if the query explicitly mentions a number
    count_pattern = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:candidate|person|applicant|interview)'
    count_match = re.search(count_pattern, scheduling_query.lower())
    explicit_count = None
    
    if count_match:
        count_text = count_match.group(1)
        # Convert text number to digit if needed
        if count_text.isdigit():
            explicit_count = int(count_text)
        else:
            text_to_num = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            explicit_count = text_to_num.get(count_text.lower())
    
    # Determine how many candidates to schedule
    if explicit_count is not None:
        # Use explicitly mentioned count, capped by available candidates
        num_to_schedule = min(explicit_count, len(resume_list))
    elif mentions_all:
        # Schedule all candidates
        num_to_schedule = len(resume_list)
    else:
        # Check if original required count exists and is valid
        if original_required_count and 1 <= original_required_count <= len(resume_list):
            # Use originally requested count
            num_to_schedule = original_required_count
        else:
            # Default to all candidates if we can't determine a specific count
            num_to_schedule = len(resume_list)
    
    # Return the appropriate candidates
    return resume_list[:num_to_schedule]

def process_bulk_scheduling(candidates_to_schedule, scheduling_query, st, llm, scheduler):
    """
    Process bulk interview scheduling with improved handling of candidate information
    
    Args:
        candidates_to_schedule: List of candidates to schedule
        scheduling_query: The user's scheduling query
        st: Streamlit instance
        llm: LLM agent instance
        scheduler: Interview scheduler instance
        
    Returns:
        Tuple of (success, response_message, interviews)
    """
    # Ensure we have candidate data properly formatted
    formatted_candidates = []
    for candidate in candidates_to_schedule:
        if isinstance(candidate, str):
            # Parse string representation
            parsed_candidate = parse_candidate_from_string(candidate)
            formatted_candidates.append(parsed_candidate)
        else:
            # Already a dictionary
            formatted_candidates.append(candidate)
    
    # Extract times from the query if mentioned
    times_pattern = r'(?:at|from)\s+(\d+(?::\d+)?(?:\s*[ap]\.?m\.?)?)'
    time_matches = re.findall(times_pattern, scheduling_query, re.IGNORECASE)
    
    # Check for "respectively" pattern which indicates specific time assignments
    is_respectively_pattern = "respectively" in scheduling_query.lower()
    
    # Convert request into format needed for scheduling
    request_data = {
        "is_bulk_scheduling": True,
        "parameters": {}
    }
    
    # Add extracted times if found
    if time_matches:
        request_data["times"] = time_matches
        
    # Extract date if mentioned
    date_pattern = r'(?:on|for)\s+(tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{2,4}?)'
    date_match = re.search(date_pattern, scheduling_query, re.IGNORECASE)
    if date_match:
        request_data["dates"] = [date_match.group(1)]
    
    # Display a note about the number of candidates being scheduled
    st.info(f"Scheduling interviews for {len(formatted_candidates)} candidates.")
    
    # Call the intelligent_interview_matching function
    interviewers = scheduler.get_available_interviewers()
    matching_result = llm.intelligent_interview_matching(formatted_candidates, interviewers)
    
    # Check if we got successful matches
    if matching_result.get("success", False) and matching_result.get("matches"):
        interviews = matching_result.get("matches", [])
        scheduling_changes = matching_result.get("changes", [])
        
        # Generate personalized invitations - one for each candidate
        invitation_drafts = {}
        try:
            # First try using the enhanced personalized invitation generator
            invitation_drafts = llm.generate_personalized_invitations(interviews)
        except Exception as invitation_e:
            # Fallback to direct generation of invitations
            logger.error(f"Error generating personalized invitations: {invitation_e}")
            for interview in interviews:
                candidate_id = interview.get("candidate_id", "unknown")
                invitation_drafts[candidate_id] = scheduler.generate_invitation_draft(interview)
        
        # Balance interviewer workload
        balanced_interviews = llm.optimize_interviewer_workload(interviews)
        
        success_message = f"Successfully scheduled {len(balanced_interviews)} interviews!"
        
        # Show scheduling changes if any occurred
        if scheduling_changes and len(scheduling_changes) > 0:
            changes_message = "Some adjustments were made to your requested schedule:"
            for change in scheduling_changes:
                changes_message += f"\n- {change.get('message', '')}"
            
            st.info(changes_message)
        
        response_message = f"I've scheduled interviews for {len(balanced_interviews)} candidates. Each candidate has been matched with the most suitable interviewer based on their skills and experience."
        
        # Add information about time adjustments if needed
        if scheduling_changes and len(scheduling_changes) > 0:
            response_message += "\n\nNote: I had to make some adjustments to your requested schedule to accommodate interviewer availability and expertise matching."
        
        return True, response_message, balanced_interviews
    else:
        # Use the schedule_multiple_interviews method with enhanced parsing
        requested_times = []
        if "times" in request_data:
            requested_times = request_data["times"]
        elif time_matches:
            requested_times = time_matches
        else:
            # Default times if none specified
            default_times = ["9am", "10am", "11am", "1pm", "2pm"]
            requested_times = default_times[:len(formatted_candidates)]
        
        # Use the date if specified, otherwise default to tomorrow
        requested_date = None
        if "dates" in request_data and request_data["dates"]:
            requested_date = request_data["dates"][0]
        
        # Prepare formatted candidates with proper name handling
        for i, candidate in enumerate(formatted_candidates):
            # Ensure candidate has a name
            if "Name" not in candidate or not candidate["Name"] or candidate["Name"] == "Unknown":
                candidate["Name"] = f"Candidate {i+1}"
        
        # Call the schedule_multiple_interviews method
        scheduled_interviews = scheduler.schedule_multiple_interviews(
            formatted_candidates,
            requested_times
        )
        
        if scheduled_interviews:
            success_message = f"Successfully scheduled {len(scheduled_interviews)} interviews!"
            response_message = f"I've scheduled interviews for {len(scheduled_interviews)} candidates. The schedule has been created based on interviewer availability and expertise matching."
            return True, response_message, scheduled_interviews
        else:
            error_message = "Could not schedule any interviews."
            response_message = "I couldn't schedule the interviews. There might be an issue with matching candidates to interviewers or with interviewer availability."
            return False, response_message, []

# Get user input via chat interface
user_query = st.chat_input("Type your message here...")

# Create sidebar with enhanced features
with st.sidebar:
    st.markdown("# Control Panel")
    st.selectbox("RAG Mode", ["RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
    model_options = ["gemini-2.0-flash", "llama-3.3-70b-versatile"]
    st.selectbox("Model", options=model_options, index=0, key="llm_selection")
    
    # Add resume filtering options
    # st.markdown("### Resume Filtering Options")
    # min_experience = st.slider("Minimum Experience (Years)", 0, 15, 0)
    # education_options = ["Any", "High School", "Associate's", "Bachelor's", "Master's", "PhD"]
    # min_education = st.selectbox("Minimum Education", options=education_options, index=0)
    
    # Add file uploader
    st.file_uploader("Upload resumes", type=["pdf"], accept_multiple_files=True, key="uploaded_files", on_change=upload_file)
    st.button("Clear conversation", on_click=clear_message)
    
    # Add evaluation metrics display in sidebar

    # Add a debug mode toggle in development environments
    if os.getenv("ENVIRONMENT", "production").lower() == "development":
        st.checkbox("Debug Mode", value=DEBUG_MODE, key="debug_mode")
        if "debug_mode" in st.session_state:
            DEBUG_MODE = st.session_state.debug_mode

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

try:
    llm = ChatBot(
        api_key=os.getenv("GROQ_API_KEY"),
        model=st.session_state.llm_selection,
    )
    logger.info(f"Initialized ChatBot with model {st.session_state.llm_selection}")
except Exception as e:
    logger.error(f"Error initializing ChatBot: {e}")
    st.error(f"Error initializing ChatBot: {str(e)}")
    # Create a minimal LLM instance as fallback
    llm = None

# Process user query
if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Create LLM if needed
    if llm is None:
        try:
            llm = ChatBot(
                api_key=os.getenv("GROQ_API_KEY"),
                model="gemini-2.0-flash",  # Fallback to a reliable model
            )
            logger.info("Created fallback ChatBot")
        except Exception as e:
            logger.error(f"Error creating fallback ChatBot: {e}")
            st.error("Unable to initialize language model. Please check your API keys.")
            st.stop()

    with st.chat_message("AI"):
        # Use enhanced_query_detection from the new code instead of simple detection functions
        try:
            query_analysis = llm.enhanced_query_detection(user_query)
            query_type = query_analysis.get("query_type", "general_question")
            is_interview_request = query_type == "interview_scheduling"
            is_bulk_scheduling = query_type == "bulk_scheduling"
            
            # Extract role, skills, and categories
            role_info = extract_role_and_skills_from_query(user_query, llm)
            categories = extract_categories_from_query(user_query)
            
            # Show debug info if enabled
            if DEBUG_MODE:
                st.write(f"Debug: resume_list length = {len(st.session_state.resume_list)}")
                st.write(f"Debug: query_type = {query_type}")
                st.write(f"Debug: is_interview_request = {is_interview_request}")
                st.write(f"Debug: is_bulk_scheduling = {is_bulk_scheduling}")
                st.write(f"Debug: query parameters = {query_analysis.get('parameters', {})}")
                st.write(f"Debug: role_info = {role_info}")
                st.write(f"Debug: categories = {categories}")
        except Exception as e:
            logger.error(f"Error in query detection: {e}")
            query_type = "general_question"
            is_interview_request = False
            is_bulk_scheduling = False
            role_info = {"role_type": "", "skills": []}
            categories = {}
            
            if DEBUG_MODE:
                st.error(f"Query detection error: {str(e)}")

        if is_bulk_scheduling:
            # Bulk interview scheduling request
            with st.spinner("Scheduling multiple interviews..."):
                try:
                    if not st.session_state.resume_list:
                        st.warning("No candidate data found. Please search for candidates first.")
                        response = ("I need candidate information to schedule interviews. "
                                "Please search for candidates first by asking something like "
                                "'find me some GenAI candidates', then request scheduling.")
                    else:
                        # Get the original required count from metadata if available
                        original_required_count = None
                        if hasattr(st.session_state, "meta_data"):
                            original_required_count = st.session_state.meta_data.get("required_count")
                        
                        # Determine which candidates to schedule based on context
                        candidates_to_schedule = determine_candidates_to_schedule(
                            st.session_state.resume_list,
                            user_query,
                            original_required_count
                        )
                        
                        # Process bulk scheduling with improved candidate handling
                        success, response, interviews = process_bulk_scheduling(
                            candidates_to_schedule,
                            user_query,
                            st,
                            llm,
                            st.session_state.interview_scheduler
                        )
                        
                        if success and interviews:
                            # Format a clean interview table for the response - improved formatting
                            interview_details = ""
                            for idx, interview in enumerate(interviews):
                                candidate_name = interview.get("candidate_id", "Unknown")
                                interviewer = interview.get("interviewer_name", "Unknown")
                                
                                # Format the date for display
                                interview_date = interview.get("date", interview.get("interview_date", ""))
                                formatted_date = interview_date
                                try:
                                    if interview_date.count("-") == 2:  # Looks like ISO format
                                        date_obj = datetime.datetime.strptime(interview_date, "%Y-%m-%d")
                                        formatted_date = date_obj.strftime("%A, %B %d, %Y")
                                except:
                                    pass
                                    
                                # Format the time for display
                                start_time = interview.get("start_time", "")
                                end_time = interview.get("end_time", "")
                                formatted_time = start_time
                                if end_time:
                                    formatted_time += f" - {end_time}"
                                
                                # Get meeting link
                                meeting_link = interview.get("meeting_link", "https://meet.google.com/" + str(uuid.uuid4())[:8])
                                
                                # Create a section for each interview with consistent formatting 
                                interview_details += f"""
                    #### {idx+1}. {candidate_name} with {interviewer}
                    - **Date:** {formatted_date}
                    - **Time:** {formatted_time}
                    - **Format:** Video Conference
                    - **Meeting Link:** [Join Meeting]({meeting_link})

                    """
                            
                            # Use the template for bulk interview scheduling
                            formatted_response = conversation_templates["bulk_interviews_scheduled"].format(
                                count=len(interviews),
                                interview_details=interview_details
                            )
                            
                            st.markdown(formatted_response)
                            
                            # Continue with showing interviews in a table format
                            st.markdown("### 📅 Interview Schedule")
                            
                            # Create a table for interviews
                            interview_data = []
                            for interview in interviews:
                                candidate_id = interview.get("candidate_id", "")
                                interviewer_name = interview.get("interviewer_name", "Unknown")
                                interview_date = interview.get("date", interview.get("interview_date", "TBD"))
                                start_time = interview.get("start_time", "TBD")
                                end_time = interview.get("end_time", "TBD") if "end_time" in interview else ""
                                
                                # Create meeting link if not provided
                                meeting_link = interview.get("meeting_link", "https://meet.google.com/" + str(uuid.uuid4())[:8])
                                
                                interview_data.append({
                                    "Candidate ID": candidate_id,
                                    "Interviewer": interviewer_name,
                                    "Date": interview_date,
                                    "Time": f"{start_time}" + (f" - {end_time}" if end_time else ""),
                                    "Meeting Link": meeting_link
                                })
                            
                            # Display as table
                            st.table(interview_data)
                            
                            # Display invitation drafts separately - one per candidate
                            st.markdown("### 📧 Email Drafts")
                            
                            # Get invitation drafts
                            invitation_drafts = {}
                            for interview in interviews:
                                candidate_id = interview.get("candidate_id", "")
                                if not candidate_id in invitation_drafts:
                                    # Generate a draft for this candidate
                                    invitation_drafts[candidate_id] = st.session_state.interview_scheduler.generate_invitation_draft(interview)
                            
                            for interview in interviews:
                                candidate_id = interview.get("candidate_id", "")
                                
                                # Get invitation draft for this candidate
                                email_draft = ""
                                if candidate_id and candidate_id in invitation_drafts:
                                    email_draft = invitation_drafts[candidate_id]
                                else:
                                    # Generate a basic invitation as fallback
                                    email_draft = st.session_state.interview_scheduler.generate_invitation_draft(interview)
                                
                                # Display in expandable section with unique key
                                with st.expander(f"Email Draft for Candidate ID: {candidate_id}", expanded=False):
                                    st.text_area(
                                        "Email Content", 
                                        email_draft, 
                                        height=200, 
                                        key=f"draft_bulk_{candidate_id}"
                                    )
                                    
                                    # Add send email button
                                    # if st.button("Send Email", key=f"send_bulk_{candidate_id}"):
                                    #     send_success = st.session_state.interview_scheduler.send_email(candidate_id)
                                    #     if send_success:
                                    #         st.session_state.email_sent_status[candidate_id] = True
                                    #         st.success("✅ Email sent successfully!")
                                    #     else:
                                    #         st.session_state.email_sent_status[candidate_id] = False
                                    #         st.error("❌ Failed to send email.")
                        else:
                            st.error("Could not schedule any interviews.")
                            st.markdown(conversation_templates["interview_error"].format(
                                error_message="Could not schedule interviews. There might be an issue with matching candidates to interviewers or with interviewer availability."
                            ))
                except Exception as e:
                    logger.error(f"Error during bulk interview scheduling: {e}")
                    st.error(f"Error during interview scheduling: {str(e)}")
                    if DEBUG_MODE:
                        import traceback
                        st.code(traceback.format_exc(), language="python")
                    response = "There was an error while scheduling the interviews. Please try again with more specific details or contact technical support if the issue persists."
        
        elif is_interview_request:
            # Process individual interview scheduling
            with st.spinner("Scheduling interview..."):
                try:
                    # Make sure we have resume data before scheduling
                    if not st.session_state.resume_list:
                        st.warning("No candidate data found. Please search for candidates first.")
                        response = "I need candidate information before I can schedule interviews. Please search for candidates first by asking about specific roles or skills."
                    else:
                        # Format the interview request data
                        request_data = {
                            "is_interview_request": True,
                            "candidate_id": query_analysis.get("parameters", {}).get("candidate_id"),
                            "time": query_analysis.get("parameters", {}).get("time", "9am"),
                            "date": query_analysis.get("parameters", {}).get("date", "tomorrow"),
                            "interviewer": query_analysis.get("parameters", {}).get("interviewer")
                        }
                        
                        # Use the enhanced schedule_interview_for_candidate method
                        scheduling_result = llm.schedule_interview_for_candidate(
                            st.session_state.interview_scheduler,
                            request_data,
                            st.session_state.resume_list
                        )
                        
                        if scheduling_result["success"]:
                            interview = scheduling_result["interview"]
                            
                            # Generate email draft
                            email_draft = scheduling_result.get("invitation_draft", "")
                            if not email_draft:
                                email_draft = st.session_state.interview_scheduler.get_email_draft(interview)
                            
                            # Create a unique key for this interview
                            candidate_id = interview.get("candidate_id", "")
                            candidate_name = interview.get("candidate_name", "Unknown")
                            interviewer_name = interview.get("interviewer_name", "Unknown")
                            
                            # Format the date for display
                            interview_date = interview.get("date", "tomorrow")
                            formatted_date = interview_date
                            try:
                                if interview_date.count("-") == 2:  # Looks like ISO format
                                    date_obj = datetime.datetime.strptime(interview_date, "%Y-%m-%d")
                                    formatted_date = date_obj.strftime("%A, %B %d, %Y")
                            except:
                                pass
                                
                            # Format the time for display
                            start_time = interview.get("start_time", "")
                            end_time = interview.get("end_time", "")
                            formatted_time = start_time
                            if end_time:
                                formatted_time += f" - {end_time}"
                            
                            # Get meeting link
                            meeting_link = interview.get("meeting_link", "")
                            
                            # Use template for single interview confirmation
                            formatted_response = conversation_templates["interview_scheduled"].format(
                                candidate_name=candidate_name,
                                interviewer_name=interviewer_name,
                                formatted_date=formatted_date,
                                formatted_time=formatted_time,
                                meeting_link=meeting_link
                            )
                            
                            st.markdown(formatted_response)
                            
                            # Display email draft in a collapsible section
                            with st.expander("📧 Email Invitation Draft", expanded=True):
                                st.text_area(
                                    "Email Content", 
                                    email_draft, 
                                    height=200, 
                                    key=f"draft_{candidate_id}"
                                )
                                
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
                            
                            # Record invitation correctness metrics
                            invitation_metrics = llm.evaluate_invitation_correctness(
                                {candidate_id: email_draft}, 
                                [interview]
                            )
                            if "overall_correctness" in invitation_metrics:
                                st.session_state.evaluation_metrics["invitation_correctness"].append(
                                    invitation_metrics["overall_correctness"]
                                )
                                
                            # Record scheduling fairness for a single interview
                            scheduling_metrics = llm.evaluate_scheduling_fairness([interview])
                            if "overall_fairness" in scheduling_metrics:
                                st.session_state.evaluation_metrics["scheduling_fairness"].append(
                                    scheduling_metrics["overall_fairness"]
                                )
                        else:
                            st.error(scheduling_result["message"])
                            # Use interview error template
                            st.markdown(conversation_templates["interview_error"].format(
                                error_message=scheduling_result["message"]
                            ))
                            response = f"I couldn't schedule the interview: {scheduling_result['message']}"
                except Exception as e:
                    logger.error(f"Error during interview scheduling: {e}")
                    st.error(f"Error during interview scheduling: {str(e)}")
                    if DEBUG_MODE:
                        import traceback
                        st.code(traceback.format_exc(), language="python")
                    response = "There was an error while scheduling the interview. Please try again with more specific details."
        else:
            # Handle general user queries (not interview requests)
            start = time.time()
            with st.spinner("Searching for candidates..."):
                try:
                    # Get required count from the query
                    required_count = extract_required_count_from_query(user_query)
                    
                    # Apply filtering from sidebar if specified
                    # filters = {}
                    # if min_experience > 0:
                    #     filters["min_experience"] = min_experience
                    # if min_education != "Any":
                    #     filters["min_education"] = min_education
                    
                    # Extract role and skills for better retrieval
                    query_info = extract_role_and_skills_from_query(user_query, llm)
                    
                    # Check for category-based query
                    if categories:
                        # Category-based query detected (e.g., "5 for GenAI, 3 for Full-stack")
                        st.info(f"Category-based query detected: {categories}")
                        
                        # Create job descriptions for each category
                        job_descriptions = {}
                        for category, count in categories.items():
                            if category == "genai":
                                job_descriptions[category] = "Generative AI Engineer with expertise in large language models, prompt engineering, and AI application development."
                            elif category == "fullstack":
                                job_descriptions[category] = "Full Stack Developer with experience in both frontend and backend technologies, building complete web applications."
                            elif category == "frontend":
                                job_descriptions[category] = "Frontend Developer with expertise in modern JavaScript frameworks, responsive design, and UI/UX implementation."
                            elif category == "backend":
                                job_descriptions[category] = "Backend Developer with strong skills in server-side programming, databases, and API development."
                            else:
                                job_descriptions[category] = f"{category.capitalize()} Developer with relevant technical skills and experience."
                        
                        # Create category-based retrieval parameters
                        job_criteria = {
                            "job_description": user_query,
                            "categories": categories
                        }
                        
                        # Use the retrieve_candidates_by_criteria method
                        document_list = retriever.retrieve_docs(
                            user_query, 
                            llm, 
                            st.session_state.rag_selection, 
                            k=sum(categories.values())
                        )
                    else:
                        # Log retrieval parameters
                        if DEBUG_MODE:
                            st.write(f"Debug: Required count = {required_count}")
                            st.write(f"Debug: Role type = {query_info.get('role_type', 'Not specified')}")
                            st.write(f"Debug: Skills = {query_info.get('skills', [])}")
                        
                        # Get documents from retriever - request more than needed to ensure we have enough
                        buffer_factor = 2  # Request twice as many to have a buffer
                        document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection, k=required_count*buffer_factor)
                    
                    # Show debug info only in debug mode
                    if DEBUG_MODE:
                        st.write(f"Debug: Retrieved {len(document_list)} documents")
                        if document_list:
                            sample_resume = document_list[0]
                            st.write(f"Debug: Sample resume format - {sample_resume[:100]}...")
                    
                    # Display warning if we don't have enough candidates
                    if categories:
                        total_requested = sum(categories.values())
                        if len(document_list) < total_requested:
                            st.warning(f"Found only {len(document_list)} candidates matching your categories, while you requested {total_requested}.")
                    elif len(document_list) < required_count:
                        st.warning(f"Found only {len(document_list)} candidates matching your query, while you requested {required_count}.")
                    
                    # Store the retrieved resumes in session state for later use (like scheduling)
                    if document_list:
                        st.session_state.resume_list = document_list
                    else:
                        st.warning("No matching resumes found in the database.")
                        # Use the no candidates found template
                        no_results_response = conversation_templates["no_candidates_found"]
                        st.markdown(no_results_response)
                        
                        # Add to chat history
                        st.session_state.chat_history.append(AIMessage(content=no_results_response))
                        st.stop()
                    
                    # Get query type from retriever metadata
                    query_type = retriever.meta_data.get("query_type", "retrieve_applicant")
                    
                    # Record shortlisting accuracy metrics
                    if hasattr(retriever, "meta_data") and "evaluation_metrics" in retriever.meta_data:
                        shortlisting_metrics = retriever.meta_data["evaluation_metrics"]
                        if "overall_accuracy" in shortlisting_metrics:
                            st.session_state.evaluation_metrics["shortlisting_accuracy"].append(
                                shortlisting_metrics["overall_accuracy"]
                            )
                    
                    # Check for shortlisting reasons and display in expanders
                    if hasattr(retriever, "meta_data") and "shortlisting_reasons" in retriever.meta_data:
                        shortlisting_reasons = retriever.meta_data["shortlisting_reasons"]
                        
                        # Create expanders for reasons if they exist
                        if shortlisting_reasons:
                            with st.expander("📋 Shortlisting Explanations", expanded=False):
                                for doc_id, reason_data in shortlisting_reasons.items():
                                    category = reason_data.get("category", "")
                                    reason = reason_data.get("reason", "No explanation available")
                                    st.markdown(f"**Candidate ID {doc_id}**:")
                                    st.markdown(reason)
                    
                    # Generate answer using the LLM with enhanced role-specific prompting
                    stream_message = llm.generate_message_stream(
                        user_query, 
                        document_list, 
                        [], 
                        query_type, 
                        required_count
                    )
                except Exception as e:
                    logger.error(f"Error retrieving or processing candidates: {e}")
                    if DEBUG_MODE:
                        import traceback
                        st.error(f"Error: {str(e)}")
                        st.code(traceback.format_exc(), language="python")
                    
                    # Create a simple error message as fallback
                    error_message = f"I encountered an error while processing your request: {str(e)}"
                    class MockStream:
                        def __iter__(self):
                            yield AIMessage(content=error_message)
                    stream_message = MockStream()
            
            end = time.time()
            
            # Display the streamed response
            try:
                response = st.write_stream(stream_message)
            except Exception as e:
                logger.error(f"Error displaying streamed response: {e}")
                response = f"I encountered an error while generating a response. Please try rephrasing your question or try again later."
                st.write(response)
            
            # Display retriever information (document summary, metadata, timing)
            try:
                retriever_message = chatbot_verbosity
                retriever_message.render(document_list, retriever.meta_data, end-start)
                
                # Add to chat history
                st.session_state.chat_history.append(AIMessage(content=response))
                st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))
            except Exception as e:
                logger.error(f"Error rendering retriever message: {e}")
                st.session_state.chat_history.append(AIMessage(content=response))
            
            # Debug info only in debug mode
            if DEBUG_MODE:
                st.write(f"Debug: resume_list length after processing = {len(st.session_state.resume_list)}")