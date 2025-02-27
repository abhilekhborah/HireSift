import sys, httpx, os
sys.dont_write_bytecode = True

from dotenv import load_dotenv

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import VoyageEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

DATA_PATH = os.getenv("GENERATED_DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RAG_K_THRESHOLD = 5
LLM_MODEL = "gemini-2.0-flash"


class ChatBot():
  def __init__(self, api_key: str, model: str):
    self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GROQ_API_KEY"))

  def generate_subquestions(self, question: str):
    system_message = SystemMessage(content="""
      You are an expert in talent acquisition. Separate this job description into 3-4 more focused aspects for efficient resume retrieval. 
      Make sure every single relevant aspect of the query is covered in at least one query. You may choose to remove irrelevant information that doesn't contribute to finding resumes such as the expected salary of the job, the ID of the job, the duration of the contract, etc.
      Only use the information provided in the initial query. Do not make up any requirements of your own.
      Put each result in one line, separated by a linebreak.
      """)
    
    user_message = HumanMessage(content=f"""
      Generate 3 to 4 sub-queries based on this initial job description: 
      {question}
    """)

    oneshot_example = HumanMessage(content="""
      Generate 3 to 4 sub-queries based on this initial job description:

      Wordpress Developer
      We are looking to hire a skilled WordPress Developer to design and implement attractive and functional websites and Portals for our Business and Clients. You will be responsible for both back-end and front-end development including the implementation of WordPress themes and plugins as well as site integration and security updates.
      To ensure success as a WordPress Developer, you should have in-depth knowledge of front-end programming languages, a good eye for aesthetics, and strong content management skills. Ultimately, a top-class WordPress Developer can create attractive, user-friendly websites that perfectly meet the design and functionality specifications of the client.
      WordPress Developer Responsibilities:
      Meeting with clients to discuss website design and function.
      Designing and building the website front-end.
      Creating the website architecture.
      Designing and managing the website back-end including database and server integration.
      Generating WordPress themes and plugins.
      Conducting website performance tests.
      Troubleshooting content issues.
      Conducting WordPress training with the client.
      Monitoring the performance of the live website.
      WordPress Developer Requirements:
      Bachelors degree in Computer Science or a similar field.
      Proven work experience as a WordPress Developer.
      Knowledge of front-end technologies including CSS3, JavaScript, HTML5, and jQuery.
      Knowledge of code versioning tools including Git, Mercurial, and SVN.
      Experience working with debugging tools such as Chrome Inspector and Firebug.
      Good understanding of website architecture and aesthetics.
      Ability to project manage.
      Good communication skills.
      Contract length: 12 months
      Expected Start Date: 9/11/2020
      Job Types: Full-time, Contract
      Salary: 12,004.00 - 38,614.00 per month
      Schedule:
      Flexible shift
      Experience:
      Wordpress: 3 years (Required)
      web designing: 2 years (Required)
      total work: 3 years (Required)
      Education:
      Bachelor's (Preferred)
      Work Remotely:
      Yes
    """)

    oneshot_response = AIMessage(content="""
      1. WordPress Developer Skills:
        - WordPress, front-end technologies (CSS3, JavaScript, HTML5, jQuery), debugging tools (Chrome Inspector, Firebug), code versioning tools (Git, Mercurial, SVN).
        - Required experience: 3 years in WordPress, 2 years in web designing.
    
      2. WordPress Developer Responsibilities:
        - Meeting with clients for website design discussions.
        - Designing website front-end and architecture.
        - Managing website back-end including database and server integration.
        - Generating WordPress themes and plugins.
        - Conducting website performance tests and troubleshooting content issues.
        - Conducting WordPress training with clients and monitoring live website performance.

      3. WordPress Developer Other Requirements:
        - Education requirement: Bachelor's degree in Computer Science or similar field.
        - Proven work experience as a WordPress Developer.
        - Good understanding of website architecture and aesthetics.
        - Ability to project manage and strong communication skills.

      4. Skills and Qualifications:
        - Degree in Computer Science or related field.
        - Proven experience in WordPress development.
        - Proficiency in front-end technologies and debugging tools.
        - Familiarity with code versioning tools.
        - Strong communication and project management abilities.
    """)

    response = self.llm.invoke([system_message, oneshot_example, oneshot_response, user_message])
    result = response.content.split("\n\n")
    return result

  def generate_message_stream(self, question: str, docs: list, history: list, prompt_cls: str):
    context = "\n\n".join(doc for doc in docs)
    
    if prompt_cls == "retrieve_applicant_jd":
      system_message = SystemMessage(content="""
        You are an expert in talent acquisition that helps determine the best candidate among multiple suitable resumes.
        Use the following pieces of context to determine the best resume given a job description. 
        You should provide some detailed explanations for the best resume choice.
        Because there can be applicants with similar names, use the applicant ID to refer to resumes in your response. 
        If you don't know the answer, just say that you don't know, do not try to make up an answer.
      """)

      user_message = HumanMessage(content=f"""
        Chat history: {history}
        Context: {context}
        Question: {question}
      """)

    else:
      system_message = SystemMessage(content="""
        You are an expert in talent acquisition that helps analyze resumes to assist resume screening.
        You may use the following pieces of context and chat history to answer your question. 
        Do not mention in your response that you are provided with a chat history.
        If you don't know the answer, just say that you don't know, do not try to make up an answer.
      """)

      user_message = HumanMessage(content=f"""
        Chat history: {history}
        Question: {question}
        Context: {context}
      """)

    stream = self.llm.stream([system_message, user_message])
    return stream

  def select_candidates_by_criteria(self, resume_list, categories):
    """
    Select candidates based on specified criteria and category counts
    
    Args:
        resume_list: List of candidate resumes
        categories: Dictionary mapping category names to number of candidates to select
        
    Returns:
        List of selected candidates with their details
    """
    system_message = SystemMessage(content=f"""
      You are an expert in talent acquisition.
      Your task is to analyze resumes and select the best candidates for each category.
      For each resume, determine which category it best fits into and assign a relevance score.
      Then select the top candidates for each category based on the specified counts.
      
      Categories and counts: {categories}
    """)
    
    user_message = HumanMessage(content=f"""
      Here are the candidate resumes:
      {resume_list}
      
      For each resume:
      1. Determine which category (GenAI or Fullstack) the candidate best fits
      2. Assign a relevance score (0-100) based on skills and experience
      3. Select the top {sum(categories.values())} candidates following the category counts
      
      Return a JSON array of objects with this structure:
      [
        {{
          "id": "candidate_id",
          "category": "category_name",
          "score": relevance_score,
          "resume": "resume_text",
          "key_skills": ["skill1", "skill2", "skill3"]
        }}
      ]
    """)
    
    response = self.llm.invoke([system_message, user_message])
    # Parse the JSON response
    try:
        # Note: In a production environment, add proper error handling for JSON parsing
        import json
        return json.loads(response.content)
    except:
        # Fallback if JSON parsing fails
        return [{"id": "parsing_error", "category": "error", "score": 0, "resume": response.content, "key_skills": []}]

  def match_interviewers(self, candidates, interviewers):
    """
    Match candidates with appropriate interviewers based on expertise and workload balance
    
    Args:
        candidates: List of selected candidates
        interviewers: List of available interviewers with expertise and availability
        
    Returns:
        List of matched interviews with scheduling details
    """
    system_message = SystemMessage(content="""
      You are an expert in talent acquisition and interview scheduling.
      Your task is to match candidates with appropriate interviewers based on expertise and workload balance.
      Prioritize matching candidates with interviewers who have relevant expertise.
      Ensure a balanced workload among interviewers - no interviewer should have significantly more interviews than others.
      Schedule interviews during the interviewers' available hours.
    """)
    
    user_message = HumanMessage(content=f"""
      Here are the candidates to be interviewed:
      {candidates}
      
      Here are the available interviewers with their expertise and availability hours (24-hour format):
      {interviewers}
      
      Match each candidate with an appropriate interviewer and assign an interview time.
      Each interview should be 45 minutes long with at least 15 minutes between interviews for the same interviewer.
      
      Return a JSON array of objects with this structure:
      [
        {{
          "candidate_id": "id",
          "candidate_category": "category",
          "interviewer_name": "name",
          "interview_date": "YYYY-MM-DD",
          "start_time": "HH:MM",
          "end_time": "HH:MM"
        }}
      ]
    """)
    
    response = self.llm.invoke([system_message, user_message])
    # Parse the JSON response
    try:
        import json
        return json.loads(response.content)
    except:
        # Fallback if JSON parsing fails
        return [{"error": "Failed to parse interviewer matching response", "details": response.content}]

  def generate_interview_invitations(self, matched_interviews):
    """
    Generate interview invitation drafts for the candidates
    
    Args:
        matched_interviews: List of interviews with candidate and interviewer details
        
    Returns:
        List of formatted interview invitation emails
    """
    system_message = SystemMessage(content="""
      You are an expert in talent acquisition and professional communication.
      Your task is to generate personalized interview invitation emails for candidates.
      Each email should include:
      1. A professional greeting
      2. Details about the interview (position, date, time, format)
      3. Information about the interviewer
      4. Next steps and preparation instructions
      5. A professional closing
    """)
    
    user_message = HumanMessage(content=f"""
      Please generate interview invitation drafts for these matched interviews:
      {matched_interviews}
      
      The interviews will be conducted via video conference.
      Candidates should prepare to discuss their experience and complete a short technical assessment.
      
      Format each invitation as a complete email.
    """)
    
    response = self.llm.invoke([system_message, user_message])
    
    # Split the response into individual invitations
    # This is a simple approach; in production, use more robust parsing
    invitations = response.content.split("\n\n---\n\n")
    if len(invitations) == 1:
        # If splitting didn't work, return the whole response as one invitation
        invitations = [response.content]
      
    return invitations

  def detect_interview_request(self, query):
    """Detect if user query is an interview scheduling request"""
    # Simple keyword-based detection
    keywords = ["schedule", "interview", "appointment", "meeting"]
    is_interview_request = any(keyword in query.lower() for keyword in keywords)
    
    if is_interview_request:
        # Parse candidate ID and time manually
        import re
        candidate_id = None
        time = None
        date = None
        
        # Try to extract candidate ID
        id_match = re.search(r'(?:Applicant|ID|candidate)\s*(?:ID)?\s*(\d+)', query, re.IGNORECASE)
        if id_match:
            candidate_id = id_match.group(1)
        
        # Try to extract time
        time_match = re.search(r'at\s+(\d+(?::\d+)?(?:\s*[ap]m)?)', query, re.IGNORECASE)
        if time_match:
            time = time_match.group(1)
            
        # Try to extract date
        date_match = re.search(r'(?:on|for)\s+(\w+day|tomorrow|today|\d+\/\d+(?:\/\d+)?)', query, re.IGNORECASE)
        date = date_match.group(1) if date_match else "tomorrow"
        
        return {
            "is_interview_request": True,
            "candidate_id": candidate_id,
            "time": time,
            "date": date
        }
    
    return {"is_interview_request": False}
  
  def detect_bulk_scheduling_request(self, query):
    """Detect if user query is requesting multiple interview scheduling"""
    # Look for patterns like "schedule interviews for X candidates" or "schedule meetings with each of them"
    keywords = ["schedule", "interview", "meeting"]
    multiple_indicators = ["each", "all", "them", "candidates", "multiple", "several"]
    
    is_scheduling_req = any(keyword in query.lower() for keyword in keywords)
    is_bulk_req = any(indicator in query.lower() for indicator in multiple_indicators)
    
    if is_scheduling_req and is_bulk_req:
        # Try to extract times
        import re
        times = re.findall(r'(\d+(?::\d+)?(?:\s*[ap]m)?)', query, re.IGNORECASE)
        
        return {
            "is_bulk_scheduling": True,
            "times": times if times else ["9am"]  # Default time if none specified
        }
    
    return {"is_bulk_scheduling": False}

  def schedule_bulk_interviews(self, scheduler, request_data, candidate_list):
    """Schedule multiple interviews based on a bulk scheduling request"""
    try:
        if not candidate_list:
            return {
                "success": False,
                "message": "No candidates found for scheduling interviews. Please search for candidates first."
            }
        
        # Extract specified times
        requested_times = request_data.get("times", ["9am"])
        
        # If fewer times than candidates, use the last time for remaining candidates
        if len(requested_times) < len(candidate_list):
            requested_times.extend([requested_times[-1]] * (len(candidate_list) - len(requested_times)))
        
        # Process candidate data if needed
        parsed_candidates = []
        for candidate in candidate_list:
            if isinstance(candidate, str):
                # Parse from string
                import re
                id_match = re.search(r'ID[:\s]*(\d+)', candidate)
                name_match = re.search(r'Name[:\s]*([^\n]+)', candidate)
                
                if id_match:
                    candidate_id = int(id_match.group(1))
                    candidate_name = name_match.group(1).strip() if name_match else f"Candidate {candidate_id}"
                    
                    # Extract skills from candidate text
                    skills = []
                    if "Python" in candidate:
                        skills.append("Python")
                    if "JavaScript" in candidate or "React" in candidate:
                        skills.append("Frontend")
                    if "Java" in candidate or "Backend" in candidate:
                        skills.append("Backend")
                    if "AI" in candidate or "ML" in candidate or "Machine Learning" in candidate:
                        skills.append("GenAI")
                    if "Full" in candidate and "Stack" in candidate:
                        skills.append("Fullstack")
                    
                    # Ensure we have at least one skill category
                    if not skills:
                        skills = ["Fullstack"]  # Default category
                    
                    parsed_candidates.append({
                        "ID": candidate_id,
                        "Name": candidate_name,
                        "Skills": skills,
                        "Resume": candidate
                    })
            else:
                # Already in dictionary format
                parsed_candidates.append(candidate)
        
        # Schedule multiple interviews
        scheduled_interviews = scheduler.schedule_multiple_interviews(
            parsed_candidates,
            requested_times[:len(parsed_candidates)]
        )
        
        # Generate invitation drafts
        invitation_drafts = scheduler.generate_bulk_invitation_drafts(scheduled_interviews)
        
        return {
            "success": True,
            "interviews": scheduled_interviews,
            "invitation_drafts": invitation_drafts,
            "message": f"Successfully scheduled {len(scheduled_interviews)} interviews."
        }
    except Exception as e:
        import traceback
        print(f"Error scheduling bulk interviews: {str(e)}")
        print(traceback.format_exc())
        return {
            "success": False,
            "message": f"Failed to schedule interviews: {str(e)}"
        }

  def schedule_interview_for_candidate(self, scheduler, request_data, resume_list):
    """Schedule an interview for a candidate"""
    try:
        # Extract information from the request
        candidate_id = request_data.get("candidate_id")
        interview_time = request_data.get("time")
        interview_date = request_data.get("date")
        
        # Check if resume_list contains strings instead of dictionaries
        if resume_list and isinstance(resume_list[0], str):
            # Parse the resume content string to extract ID and other information
            parsed_resumes = []
            for resume_str in resume_list:
                # Try to find ID pattern in the resume text
                import re
                id_match = re.search(r'ID[:\s]*(\d+)', resume_str)
                name_match = re.search(r'Name[:\s]*([^\n]+)', resume_str)
                
                if id_match:
                    resume_id = int(id_match.group(1))
                    resume_name = name_match.group(1).strip() if name_match else f"Candidate {resume_id}"
                    
                    # Extract skills from resume text for better interviewer matching
                    skills = []
                    if "Python" in resume_str:
                        skills.append("Python")
                    if "JavaScript" in resume_str or "React" in resume_str:
                        skills.append("Frontend")
                    if "Java" in resume_str or "Backend" in resume_str:
                        skills.append("Backend")
                    if "AI" in resume_str or "ML" in resume_str or "Machine Learning" in resume_str:
                        skills.append("GenAI")
                    if "Full" in resume_str and "Stack" in resume_str:
                        skills.append("Fullstack")
                    
                    # Ensure we have at least one skill category
                    if not skills:
                        skills = ["Fullstack"]  # Default category
                    
                    parsed_resumes.append({
                        "ID": resume_id,
                        "Name": resume_name,
                        "Skills": skills,
                        "Resume": resume_str
                    })
            
            # Replace the original list with the parsed version
            resume_list = parsed_resumes
        
        # If we don't have a candidate ID but have resume list, use the first one
        if candidate_id is None and resume_list:
            if isinstance(resume_list[0], dict):
                candidate_id = str(resume_list[0].get("ID", 0))
                candidate_name = resume_list[0].get("Name", f"Candidate {candidate_id}")
                candidate_expertise = resume_list[0].get("Skills", ["Fullstack"])
            else:
                # If it's still not a dictionary (shouldn't happen after parsing)
                print(f"Warning: resume_list[0] is still not a dictionary after parsing, it's a {type(resume_list[0])}")
                candidate_id = "0"
                candidate_name = "Unknown Candidate"
                candidate_expertise = ["Fullstack"]
        else:
            # If we have a specific candidate ID, find that candidate
            candidate_found = False
            for resume in resume_list:
                if isinstance(resume, dict) and str(resume.get("ID")) == str(candidate_id):
                    candidate_name = resume.get("Name", f"Candidate {candidate_id}")
                    candidate_expertise = resume.get("Skills", ["Fullstack"])
                    candidate_found = True
                    break
            
            if not candidate_found:
                candidate_name = f"Candidate {candidate_id}"
                candidate_expertise = ["Fullstack"]  # Default expertise
        
        # If we still don't have a candidate ID, return error
        if candidate_id is None:
            return {
                "success": False,
                "message": "Could not identify a candidate for the interview. Please specify a candidate ID or search for candidates first."
            }
        
        # Convert candidate_id to integer if it's a string
        if isinstance(candidate_id, str):
            candidate_id = int(candidate_id)
            
        # Schedule the interview
        interview = scheduler.schedule_interview(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            candidate_expertise=candidate_expertise,
            requested_time=interview_time,
            requested_date=interview_date
        )
        
        return {
            "success": True,
            "interview": interview,
            "message": f"Successfully scheduled an interview with {candidate_name} for {interview.date} at {interview.start_time}."
        }
    except Exception as e:
        import traceback
        print(f"Error scheduling interview: {str(e)}")
        print(traceback.format_exc())
        return {
            "success": False,
            "message": f"Failed to schedule interview: {str(e)}"
        }

    