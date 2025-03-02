import sys, httpx, os, re, json
from datetime import datetime, timedelta
sys.dont_write_bytecode = True

from dotenv import load_dotenv

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import VoyageEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import random
import numpy as np
from collections import defaultdict


load_dotenv()

DATA_PATH = os.getenv("GENERATED_DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RAG_K_THRESHOLD = 5
LLM_MODEL = "gemini-2.0-flash"


class ChatBot():
  def __init__(self, api_key: str, model: str):
    self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GROQ_API_KEY"))
    # Skills taxonomy for related skills matching
    self.skills_taxonomy = {
        "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php"],
        "frontend": ["react", "angular", "vue", "html", "css", "bootstrap", "tailwind"],
        "backend": ["django", "flask", "express", "spring", "node.js", "laravel", "fastapi"],
        "database": ["sql", "mysql", "postgresql", "mongodb", "nosql", "sqlite", "oracle", "redis"],
        "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "serverless"],
        "ai": ["machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "genai"],
        "mobile": ["android", "ios", "react native", "flutter", "swift", "kotlin"],
        "devops": ["ci/cd", "jenkins", "github actions", "gitlab ci", "ansible", "chef", "puppet"],
        "testing": ["unit testing", "integration testing", "selenium", "jest", "pytest", "cypress"],
        "blockchain": ["ethereum", "solidity", "web3", "bitcoin", "hyperledger", "smart contracts"]
    }
    
    # Reverse mapping for quick lookup
    self.skill_to_category = {}
    for category, skills in self.skills_taxonomy.items():
        for skill in skills:
            self.skill_to_category[skill] = category
    
    # Initialize metrics for evaluation
    self.evaluation_metrics = {
        "shortlisting_accuracy": 0,
        "contact_extraction_efficiency": 0,
        "invitation_correctness": 0,
        "scheduling_fairness": 0
    }

  def analyze_job_description(self, job_description: str):
    """
    Extract key information from a job description
    
    Args:
        job_description: The job description text
        
    Returns:
        Dictionary with extracted information
    """
    system_message = SystemMessage(content="""
        You are an expert in technical recruitment and job analysis.
        Extract key requirements and skills from job descriptions.
        Separate required skills from preferred skills.
        Identify minimum experience and education requirements.
        Focus only on the factual information present in the job description.
        Provide structured output as requested.
    """)
    
    user_message = HumanMessage(content=f"""
        Analyze this job description and extract the following information:
        
        {job_description}
        
        Return a JSON object with:
        1. role_title: The job title
        2. required_skills: List of required technical skills
        3. preferred_skills: List of preferred but not required skills
        4. min_experience_years: Minimum years of experience required
        5. education_level: Minimum education level required
        6. role_category: Main category (e.g., "fullstack", "genai", "backend", "frontend", etc.)
        
        Return ONLY the JSON object, no other text.
    """)
    
    try:
        response = self.llm.invoke([system_message, user_message])
        result = json.loads(response.content)
        return result
    except Exception as e:
        print(f"Error analyzing job description: {e}")
        # Fallback to simple extraction
        return self._extract_job_info_fallback(job_description)

  def _extract_job_info_fallback(self, job_description):
    """Simple fallback extraction for job descriptions when LLM fails"""
    # Extract role title
    title_match = re.search(r"(?:looking for|hiring|for the role of|position for)([\w\s]+?)(?:to join|role|position)", job_description)
    role_title = title_match.group(1).strip() if title_match else "Software Engineer"
    
    # Extract skill keywords
    skill_keywords = [
        "python", "java", "javascript", "react", "angular", "vue", "node", "django", "flask",
        "aws", "azure", "gcp", "docker", "kubernetes", "sql", "nosql", "mongodb", "postgresql",
        "machine learning", "deep learning", "ai", "genai", "frontend", "backend", "fullstack"
    ]
    
    # Simple skill extraction
    found_skills = []
    for skill in skill_keywords:
        if skill.lower() in job_description.lower():
            found_skills.append(skill)
    
    # Assume 60% required, 40% preferred
    required_count = max(1, int(len(found_skills) * 0.6))
    required_skills = found_skills[:required_count]
    preferred_skills = found_skills[required_count:]
    
    # Extract experience
    exp_match = re.search(r"(\d+)\+?\s*years", job_description.lower())
    min_experience_years = int(exp_match.group(1)) if exp_match else 1
    
    # Extract education
    edu_levels = ["bachelor", "master", "phd", "doctorate", "high school"]
    education_level = "Bachelor's"
    for level in edu_levels:
        if level in job_description.lower():
            if level == "phd" or level == "doctorate":
                education_level = "PhD"
            elif level == "master":
                education_level = "Master's"
            elif level == "high school":
                education_level = "High School"
            break
    
    # Determine role category
    if "genai" in job_description.lower() or "generative ai" in job_description.lower():
        role_category = "genai"
    elif "full stack" in job_description.lower() or "fullstack" in job_description.lower():
        role_category = "fullstack"
    elif "front" in job_description.lower():
        role_category = "frontend"
    elif "back" in job_description.lower():
        role_category = "backend"
    else:
        role_category = "software"
    
    return {
        "role_title": role_title,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "min_experience_years": min_experience_years,
        "education_level": education_level,
        "role_category": role_category
    }

  def generate_subquestions(self, question: str):
    """
    Generate focused subquestions from a main query to improve retrieval
    
    Args:
        question: The main user query
        
    Returns:
        List of focused subquestions
    """
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

    try:
        response = self.llm.invoke([system_message, oneshot_example, oneshot_response, user_message])
        result = response.content.split("\n\n")
        return result
    except Exception as e:
        print(f"Error generating subquestions: {e}")
        # Provide fallback subquestions based on the original query
        return [
            f"{question} - skills and experience",
            f"{question} - technical qualifications",
            f"{question} - education and background"
        ]

  def generate_message_stream(self, question: str, docs: list, history: list, prompt_cls: str, required_count=None):
    """
    Generate a streaming response to answer the user's query based on retrieved documents
    with enhanced category support
    
    Args:
        question: User query string
        docs: Retrieved documents/resumes
        history: Conversation history
        prompt_cls: Type of prompt to use
        required_count: Number of candidates required (defaults to None)
        
    Returns:
        Streaming response from the LLM
    """
    context = "\n\n".join(doc for doc in docs)
    
    # Extract role and skills from the query
    role_info = self._extract_role_and_skills(question)
    role_type = role_info.get("role_type", "")
    skills = role_info.get("skills", [])
    
    # Check if we have a category-based query by looking for Category tags
    is_category_query = False
    categories = {}
    
    for doc in docs:
        category_match = re.search(r'Category:\s+(\w+)', doc)
        if category_match:
            is_category_query = True
            category = category_match.group(1)
            if category not in categories:
                categories[category] = []
            
            # Extract ID to add to the category list
            id_match = re.search(r'Applicant ID\s+(\d+)', doc)
            if id_match:
                categories[category].append(id_match.group(1))
    
    if is_category_query:
        # Category-based prompt
      system_message = SystemMessage(content=f"""
      You are an expert in talent acquisition specializing in candidate assessment and role matching.

      Use the provided context to analyze and summarize candidates by job category. Your analysis should be comprehensive, data-driven, and structured for maximum clarity.

      For EACH candidate:
      1. Present a detailed breakdown of their technical skills and domain experience relevant to their specific job category
      2. Analyze their qualifications against industry standards and the specific needs of the role
      3. Provide evidence-based reasoning for why they are a strong match for their category
      4. Always include their Applicant ID in a clear, consistent format for easy reference

      Structure your response with the following hierarchy:
      - Begin with a brief introduction summarizing the overall candidate pool
      - Create distinct, clearly labeled sections for EACH job category
      - Start each category section with a header formatted as: "## [CATEGORY NAME] (x candidates)" where x is the exact number of candidates in that category
      - Within each category, present candidates in order of their apparent strength of fit
      - Use consistent formatting for each candidate profile for easy scanning

      IMPORTANT REQUIREMENTS:
      1. Group candidates ONLY by their explicitly assigned categories ({", ".join(categories.keys())})
      2. For each category, provide substantive explanations that demonstrate deep understanding of industry requirements
      3. Use markdown formatting to enhance readability with headers, bullet points, and emphasis where appropriate
      4. Ensure every candidate summary includes their Applicant ID in the format: [ID: XXX-XXX]
      5. If a category has no candidates, explicitly state this rather than omitting the section

      If the provided context contains insufficient information to assess a candidate properly, acknowledge the information gap specifically rather than making assumptions.
      """)

      user_message = HumanMessage(content=f"""
      Chat history: {history}
      Context: {context}
      Question: {question}
      REQUIREMENTS:
      1. Group candidates by their categories ({", ".join(categories.keys())})
      2. For each category, present the candidates with clear explanations
      3. Format your response with clear sections for each category
      4. Make sure to include the Applicant ID for each candidate
      """)
    
    else:
        # Standard response for job description query
        system_message = SystemMessage(content=f"""
            You are an expert in talent acquisition that helps determine the best candidates among multiple suitable resumes.
            Use the following pieces of context to identify {required_count or 'the'} candidates that best match the job description.
            
            You should provide detailed explanations for each candidate choice, including:
            - Their relevant technical skills and experience
            - How they match the specific requirements 
            - Any standout qualifications they possess
            
            Because there can be applicants with similar names, always use the applicant ID to refer to resumes in your response.
            
            IMPORTANT: You must provide exactly {required_count or 'the requested number of'} candidates in your response, unless there are fewer than 
            {required_count or 'requested'} candidates in the provided context. If there are fewer candidates available, clearly state this 
            limitation while still providing detailed analyses of all available candidates.
            
            For each candidate, summarize:
            1. Skills match: How well the candidate's skills align with the job requirements
            2. Experience match: How relevant their work experience is for the role
            3. Education match: Whether they meet the educational requirements
            4. Overall assessment: Why they would be a good fit for the role
            
            If you don't know the answer, just say that you don't know, do not try to make up an answer.
        """)

        user_message = HumanMessage(content=f"""
            Chat history: {history}
            Context: {context}
            Question: {question}
            
            REQUIREMENTS:
            1. Identify {required_count or 'the requested number of'} candidates that best match the job description
            2. If the role type is specified as "{role_type}", focus on skills relevant to that role type
            3. Prioritize candidates with these specific skills if mentioned: {', '.join(skills) if skills else 'Any relevant skills'}
            4. Format your response with clear sections for each candidate
            5. Provide a detailed explanation of why each candidate is a good match
            6. If you cannot find enough candidates, clearly state how many you found out of how many were requested
        """)

    # If fewer docs than required, add a note
    if not is_category_query and required_count is not None and len(docs) < required_count:
        additional_note = f"""
        IMPORTANT NOTE: There are only {len(docs)} candidates available in the database, but {required_count} were requested.
        Please acknowledge this limitation in your response while still providing detailed information about the available candidates.
        """
        user_message.content += additional_note

    try:
        stream = self.llm.stream([system_message, user_message])
        return stream
    except Exception as e:
        print(f"Error generating message stream: {e}")
        # Return a simple error message if streaming fails
        return [AIMessage(content=f"I encountered an error while processing your request. Please try again or modify your query. Error details: {str(e)}")]

  def _extract_role_and_skills(self, query):
    """
    Extract role type and skills from a user query
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with role_type and skills
    """
    role_type = ""
    skills = []
    
    # Extract role type with pattern matching
    role_patterns = [
        r"for (?:a|an) ([\w\s]+?) (?:role|position|job)",
        r"([\w\s]+?) developer",
        r"([\w\s]+?) engineer",
        r"([\w\s]+?) designer",
        r"([\w\s]+?) architect",
        r"([\w\s]+?) analyst"
    ]
    
    for pattern in role_patterns:
        match = re.search(pattern, query.lower())
        if match:
            role_type = match.group(1).strip()
            break
    
    # Extract skills using common technical keywords
    common_skills = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php",
        "react", "angular", "vue", "node", "express", "django", "flask", "spring",
        "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "terraform",
        "sql", "mysql", "postgresql", "mongodb", "nosql", "redis",
        "machine learning", "ai", "data science", "nlp", "computer vision",
        "frontend", "backend", "fullstack", "full-stack", "full stack", "mobile", "ios", "android",
        "devops", "cicd", "jenkins", "agile", "scrum"
    ]
    
    for skill in common_skills:
        if skill in query.lower():
            skills.append(skill)
    
    return {
        "role_type": role_type,
        "skills": skills
    }

  def select_candidates_by_criteria(self, resume_list, categories):
    """
    Select candidates based on specified criteria and category counts using semantic understanding
    
    Args:
        resume_list: List of candidate resumes
        categories: Dictionary mapping category names to number of candidates to select
        
    Returns:
        List of selected candidates with their details
    """
    system_message = SystemMessage(content=f"""
      You are an expert in talent acquisition with deep knowledge of technical roles.
      Analyze each resume to understand the candidate's skills, experience, and background comprehensively.
      
      Your task:
      1. Determine which category best fits each candidate based on their full skill profile and experience
      2. Consider both explicit skills mentioned and implicit skills suggested by their experience
      3. Calculate a relevance score that accounts for: years of experience, skill relevance, project complexity, and education
      4. Make selections that create a diverse but highly qualified candidate pool
      
      Categories and counts: {categories}
    """)
    
    user_message = HumanMessage(content=f"""
      Here are the candidate resumes:
      {resume_list}
      
      For each resume:
      1. Analyze the complete resume content to extract both stated and implied skills
      2. Map the candidate to their best-fit category: {list(categories.keys())}
      3. Calculate a comprehensive relevance score (0-100) considering depth of experience and skill match
      4. Select the top {sum(categories.values())} candidates following the category distribution
      
      Return a JSON array with this structure:
      [
        {{
          "id": "candidate_id",
          "category": "category_name",
          "score": relevance_score,
          "resume": "resume_text",
          "key_skills": ["skill1", "skill2", "skill3"],
          "years_experience": number,
          "education": "education_level"
        }}
      ]
    """)
    
    try:
        response = self.llm.invoke([system_message, user_message])
        # Parse the JSON response
        return json.loads(response.content)
    except Exception as e:
        print(f"Error in select_candidates_by_criteria: {e}")
        # Fallback if JSON parsing fails
        return [{"id": "parsing_error", "category": "error", "score": 0, "resume": str(e), "key_skills": []}]

  def match_interviewers(self, candidates, interviewers):
    """
    Match candidates with appropriate interviewers using an intelligent algorithm that 
    balances expertise matching, workload distribution, and availability constraints
    
    Args:
        candidates: List of selected candidates
        interviewers: List of available interviewers with expertise and availability
        
    Returns:
        List of matched interviews with scheduling details
    """
    system_message = SystemMessage(content="""
      You are an advanced AI scheduling assistant specialized in optimizing interview assignments.
      Your task is to create the best possible match between candidates and interviewers with these priorities:
      
      1. EXPERTISE MATCH: The interviewer's expertise should align closely with the candidate's skills
      2. WORKLOAD BALANCE: Distribute interviews evenly among available interviewers
      3. AVAILABILITY OPTIMIZATION: Schedule efficiently within available time slots
      4. INTERVIEW QUALITY: Consider time buffer between interviews and interviewer fatigue
      5. DIVERSITY OF PERSPECTIVE: When possible, assign candidates to different interviewers
      
      Apply a hybrid approach that combines constraint satisfaction and optimization algorithms in your reasoning.
    """)
    
    user_message = HumanMessage(content=f"""
      CANDIDATE DATA:
      {candidates}
      
      INTERVIEWER DATA:
      {interviewers}
      
      SCHEDULING CONSTRAINTS:
      - Each interview should last 45 minutes
      - Minimum of 15 minutes between interviews for the same interviewer
      - Interviews can only be scheduled during interviewer's available hours
      - Maximum of 5 interviews per interviewer per day to prevent fatigue
      - If possible, expertise in at least one of the candidate's key skills
      
      SCHEDULING OBJECTIVES:
      - Minimize variance in interview count among interviewers
      - Maximize expertise match between interviewer and candidate
      - Maintain sufficient breaks for interviewers with multiple interviews
      
      OUTPUT FORMAT:
      Return a JSON array of interview assignments with this structure:
      [
        {{
          "candidate_id": "id",
          "candidate_name": "name",
          "candidate_category": "category",
          "candidate_key_skills": ["skill1", "skill2"],
          "interviewer_name": "name",
          "interviewer_expertise": ["expertise1", "expertise2"],
          "expertise_match_score": float,
          "interview_date": "YYYY-MM-DD",
          "start_time": "HH:MM",
          "end_time": "HH:MM"
        }}
      ]
    """)
    
    try:
        response = self.llm.invoke([system_message, user_message])
        # Parse the JSON response
        return json.loads(response.content)
    except Exception as e:
        print(f"Error in match_interviewers: {e}")
        # Fallback if JSON parsing fails
        return [{"error": "Failed to parse interviewer matching response", "details": str(e)}]

  def generate_interview_invitations(self, matched_interviews):
    """
    Generate personalized and engaging interview invitation drafts for candidates
    
    Args:
        matched_interviews: List of interviews with candidate and interviewer details
        
    Returns:
        Dictionary mapping candidate IDs to personalized invitation texts
    """
    system_message = SystemMessage(content="""
      You are an expert in talent acquisition communications with excellent writing skills.
      Create personalized interview invitation emails that are:
      
      1. PROFESSIONAL: Maintain a professional but warm tone throughout
      2. PERSONALIZED: Reference the candidate's specific background and skills
      3. INFORMATIVE: Include all essential details about the interview
      4. CLEAR: Provide explicit instructions for next steps
      5. ENGAGING: Generate excitement about the opportunity
      6. BRANDED: Communicate company values implicitly
      
      Each email should feel like it was written specifically for that candidate.
    """)
    
    user_message = HumanMessage(content=f"""
      Please generate personalized interview invitation drafts for these matched interviews:
      {matched_interviews}
      
      For each invitation, include:
      1. Personalized greeting using candidate's name
      2. Position-specific introduction that references candidate's relevant experience
      3. Complete interview details (date, time, format, interviewer name and role)
      4. Preparation instructions including topics that will be covered
      5. Technical setup instructions for the video conference
      6. Clear next steps and contact information for questions
      7. Professional closing
      
      The company values innovation, collaboration, and technical excellence. 
      Convey enthusiasm about the candidate's potential fit with our team.
    """)
    
    try:
        response = self.llm.invoke([system_message, user_message])
        
        # Process the response into individual invitations
        invitation_dict = {}
        
        # Try to detect if the response is in JSON format first
        try:
            json_response = json.loads(response.content)
            if isinstance(json_response, list):
                for invitation in json_response:
                    candidate_id = invitation.get("candidate_id", "")
                    invitation_text = invitation.get("email_draft", "")
                    invitation_dict[candidate_id] = invitation_text
                return invitation_dict
        except json.JSONDecodeError:
            pass
        
        # Try to detect if the response contains multiple invitations
        email_pattern = r"Subject:.*?(?=Subject:|$)"
        email_matches = re.findall(email_pattern, response.content, re.DOTALL)
        
        if email_matches:
            # Now extract candidate IDs from each email draft
            for email_draft in email_matches:
                # Try to find candidate ID in the draft
                id_pattern = r"(?:ID|id|ID:|id:|ID#|id#)[\s:]*(\d+)"
                id_match = re.search(id_pattern, email_draft)
                if id_match:
                    candidate_id = id_match.group(1)
                    invitation_dict[candidate_id] = email_draft.strip()
                else:
                    # Try to match by name
                    name_pattern = r"Dear\s+([^,\n]+)"
                    name_match = re.search(name_pattern, email_draft)
                    if name_match:
                        candidate_name = name_match.group(1).strip()
                        # Find the interview with this candidate name
                        for interview in matched_interviews:
                            if interview.get("candidate_name", "") == candidate_name:
                                candidate_id = interview.get("candidate_id", "")
                                if candidate_id:
                                    invitation_dict[candidate_id] = email_draft.strip()
                                    break
        else:
            # Alternative approach: Split by candidate or standard separators
            separators = ["---", "===", "***", "\n\n\n", "Dear"]
            for separator in separators:
                if separator in response.content:
                    if separator == "Dear" and response.content.count("Dear") > 1:
                        # Split by "Dear" but include "Dear" in each part
                        parts = response.content.split("Dear")
                        invitations = ["Dear" + part for part in parts[1:] if part.strip()]
                        
                        # Try to match each invitation with a candidate
                        for idx, interview in enumerate(matched_interviews):
                            if idx < len(invitations):
                                candidate_id = interview.get("candidate_id", "")
                                if candidate_id:
                                    invitation_dict[candidate_id] = invitations[idx].strip()
                        
                        if invitation_dict:
                            break
                    else:
                        parts = response.content.split(separator)
                        invitations = [part.strip() for part in parts if part.strip()]
                        
                        # Try to match each invitation with a candidate
                        for idx, interview in enumerate(matched_interviews):
                            if idx < len(invitations):
                                candidate_id = interview.get("candidate_id", "")
                                if candidate_id:
                                    invitation_dict[candidate_id] = invitations[idx].strip()
                        
                        if invitation_dict:
                            break
        
        # If we still don't have invitation drafts, create fallbacks
        if not invitation_dict:
            for interview in matched_interviews:
                candidate_id = interview.get("candidate_id", "")
                if candidate_id:
                    invitation_dict[candidate_id] = self._generate_fallback_invitation(
                        interview.get("candidate_name", "Candidate"),
                        interview.get("interview_date", "tomorrow"),
                        interview.get("start_time", "9:00 AM"),
                        interview.get("end_time", "9:45 AM"),
                        interview.get("interviewer_name", "Interviewer"),
                        interview.get("meeting_link", "https://meet.google.com/")
                    )
        
        # Evaluate invitation correctness
        self.evaluate_invitation_correctness(invitation_dict, matched_interviews)
        
        return invitation_dict
    except Exception as e:
        print(f"Error generating interview invitations: {e}")
        # Create fallback invitations
        invitation_dict = {}
        for interview in matched_interviews:
            candidate_id = interview.get("candidate_id", "")
            if candidate_id:
                invitation_dict[candidate_id] = self._generate_fallback_invitation(
                    interview.get("candidate_name", "Candidate"),
                    interview.get("interview_date", "tomorrow"),
                    interview.get("start_time", "9:00 AM"),
                    interview.get("end_time", "9:45 AM"),
                    interview.get("interviewer_name", "Interviewer"),
                    interview.get("meeting_link", "https://meet.google.com/")
                )
        return invitation_dict

  def _generate_fallback_invitation(self, candidate_name, date, start_time, end_time, interviewer_name, meeting_link):
      """Generate a simple template-based invitation as fallback"""
      invitation = f"""
Subject: Interview Invitation

Dear {candidate_name},

We are pleased to invite you to interview at our company. After reviewing your application, we believe your skills and experience align well with what we're looking for.

Your interview has been scheduled as follows:
• Date: {date}
• Time: {start_time} - {end_time}
• Format: Video Conference
• Meeting Link: {meeting_link}
• Interviewer: {interviewer_name}

Please prepare by reviewing your previous projects and being ready to discuss your technical experience in detail. If you need to reschedule or have any questions, please reply to this email.

We look forward to meeting you and learning more about your background and interests.

Best regards,
Hiring Team
      """
      
      return invitation.strip()

  def enhanced_query_detection(self, query):
    """
    Advanced query detection with improved error handling and fallbacks
    
    Args:
        query: User's natural language query
        
    Returns:
        Dictionary with query type and parameters
    """
    try:
        # First try pattern-based detection as it's more reliable
        detection_result = self._pattern_based_detection(query)
        if detection_result.get("confidence", 0) > 0.7:
            return detection_result
        
        # If pattern detection isn't confident, try LLM-based detection
        # First, use the LLM for intent classification with few-shot examples
        system_message = SystemMessage(content="""
          You are an expert in natural language understanding for HR and recruitment systems.
          Analyze user queries to detect intent and extract relevant parameters for a recruitment assistant.
          Identify if the query is about:
          1. Candidate search
          2. Single interview scheduling
          3. Bulk interview scheduling 
          4. Interview invitation customization
          5. Interview process question
          6. Workload distribution
          7. General HR question
          
          Pay special attention to scheduling requests. If a query mentions scheduling meetings/interviews 
          with multiple candidates or uses plural terms like "them", "all", "each", "everyone", "everybody", it is likely 
          a bulk scheduling request.
          
          For bulk scheduling, look for specific time assignments patterns like "9am and 10am respectively"
          which indicates specific time slots for specific candidates.
          
          For search queries, extract:
          - Required count (default to 3 if not specified)
          - Role type (e.g., "fullstack developer", "data scientist")
          - Skills mentioned
        """)
        
        few_shot_examples = [
          {"query": "Find me developers who know Python and React", 
          "type": "candidate_search", 
          "parameters": {"skills": ["Python", "React"], "required_count": 3, "role_type": "developer"}},
          
          {"query": "Schedule an interview with candidate #1234 tomorrow at 2pm", 
          "type": "interview_scheduling", 
          "parameters": {"candidate_id": "1234", "date": "tomorrow", "time": "2pm"}},
          
          {"query": "Set up interviews for all the Java developers", 
          "type": "bulk_scheduling", 
          "parameters": {"skill_filter": "Java", "date": "tomorrow"}},
          
          {"query": "please schedule a meet with each of them depending on available slots for thursday", 
          "type": "bulk_scheduling", 
          "parameters": {"date": "thursday"}},
          
          {"query": "please schedule interviews with both of them on thursday at 9am and 10am respectively", 
          "type": "bulk_scheduling", 
          "parameters": {"date": "thursday", "times": ["9am", "10am"], "pattern": "respectively"}},
          
          {"query": "find me 5 candidates for a data scientist role", 
          "type": "candidate_search", 
          "parameters": {"required_count": 5, "role_type": "data scientist"}},
          
          {"query": "show three fullstack developers", 
          "type": "candidate_search", 
          "parameters": {"required_count": 3, "role_type": "fullstack developer"}}
        ]
        
        user_message = HumanMessage(content=f"""
          Based on this user query, determine the intent and extract relevant parameters:
          "{query}"
          
          Return the result in JSON format with these fields:
          - type: The query type (candidate_search, interview_scheduling, bulk_scheduling, invitation_customization, process_question, workload_distribution, general_question)
          - confidence: Your confidence in this classification (0-1)
          - parameters: Any relevant parameters extracted from the query
          
          For candidate search, extract:
          - required_count: Number of candidates requested (default to 3 if not specified)
          - role_type: The job role mentioned (e.g. "developer", "engineer")
          - skills: List of specific skills mentioned
          
          For bulk scheduling requests with multiple times, extract:
          - date: When the interviews should be scheduled
          - times: List of all mentioned times
          - pattern: "respectively" if times should be assigned in order to the candidates
        """)
        
        response = self.llm.invoke([system_message, user_message])
        
        # Try to parse the response as JSON
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If JSON parsing fails, extract using regex
            print("LLM didn't return valid JSON, extracting with regex...")
            
            # Extract the result type from the response
            type_pattern = r'type["\s:]+([a-z_]+)'
            type_match = re.search(type_pattern, response.content)
            query_type = type_match.group(1) if type_match else "general_question"
            
            # Create a default result
            result = {
                "type": query_type,
                "confidence": 0.5,
                "parameters": {}
            }
            
            # Extract parameters if we can
            parameters_pattern = r'parameters["\s:]+({[^}]+})'
            parameters_match = re.search(parameters_pattern, response.content)
            if parameters_match:
                try:
                    parameters_str = parameters_match.group(1)
                    # Clean up the parameters string for parsing
                    parameters_str = parameters_str.replace("'", '"')
                    parameters = json.loads(parameters_str)
                    result["parameters"] = parameters
                except:
                    pass
        
        # Enhanced detection for "respectively" pattern
        if result.get("type") == "bulk_scheduling" and "respectively" in query.lower():
            if "parameters" not in result:
                result["parameters"] = {}
            result["parameters"]["pattern"] = "respectively"
            
            # Extract times if not already done
            if "times" not in result["parameters"]:
                time_pattern = r'(?:at|from)\s+(\d+(?::\d+)?(?:\s*[ap]\.?m\.?)?)'
                time_matches = re.findall(time_pattern, query, re.IGNORECASE)
                if time_matches:
                    result["parameters"]["times"] = time_matches
        
        # Add default required_count for candidate search if not specified
        if result.get("type") == "candidate_search":
            if "parameters" not in result:
                result["parameters"] = {}
            if "required_count" not in result["parameters"]:
                result["parameters"]["required_count"] = 3
        
        # Compare LLM result with pattern-based detection
        if result.get("confidence", 0) < detection_result.get("confidence", 0):
            # Use pattern detection if it's more confident
            return detection_result
        
        return {
            "query_type": result.get("type", "general_question"),
            "confidence": result.get("confidence", 0),
            "parameters": result.get("parameters", {})
        }
    except Exception as e:
        print(f"Error in query detection: {str(e)}")
        # Always fall back to pattern-based detection
        return self._pattern_based_detection(query)

  def _pattern_based_detection(self, query):
    """
    Reliable pattern-based query detection that doesn't depend on the LLM
    
    Args:
        query: User's query string
        
    Returns:
        Dictionary with query type and parameters
    """
    query_lower = query.lower()
    
    # Check for scheduling-related terms
    scheduling_terms = ["schedule", "set up", "arrange", "book", "plan", "organize", "appointment"]
    meet_terms = ["interview", "meeting", "meet", "session", "talk to", "speak with", "call"]
    
    has_scheduling_intent = any(term in query_lower for term in scheduling_terms) or any(term in query_lower for term in meet_terms)
    
    # Check for bulk indicators
    bulk_indicators = ["all", "them", "each", "everyone", "candidates", "multiple", "both"]
    multiple_mentions = query_lower.count("candidate") > 1 or query_lower.count("interview") > 1
    
    # Check for "respectively" pattern
    has_respectively = "respectively" in query_lower
    
    # If scheduling intent is detected
    if has_scheduling_intent:
        # Determine if it's bulk or single scheduling
        if any(indicator in query_lower for indicator in bulk_indicators) or multiple_mentions:
            # Extract parameters for bulk scheduling
            parameters = {}
            
            # Look for date
            date_pattern = r'(?:on|for)\s+(tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{2,4}?)'
            date_match = re.search(date_pattern, query_lower)
            if date_match:
                parameters["date"] = date_match.group(1)
                
            # Look for time ranges
            time_pattern = r'(?:at|from)\s+(\d+(?::\d+)?(?:\s*[ap]\.?m\.?)?)'
            time_matches = re.findall(time_pattern, query_lower)
            if time_matches:
                parameters["times"] = time_matches
            
            # Check for "respectively" pattern
            if has_respectively:
                parameters["pattern"] = "respectively"
            
            # Extract number of candidates to schedule
            count_patterns = [
                r'(\d+)\s+of\s+them',
                r'schedule\s+(\d+)',
                r'for\s+(\d+)\s+candidates'
            ]
            
            for pattern in count_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    try:
                        parameters["count"] = int(match.group(1))
                        break
                    except ValueError:
                        pass
            
            return {"query_type": "bulk_scheduling", "confidence": 0.9, "parameters": parameters}
        else:
            # Extract parameters for single interview
            parameters = {}
            
            # Find candidate ID
            id_pattern = r'(?:(?:candidate|applicant|id)[\s:]*#?\s*(\d+))|(?:#\s*(\d+))'
            id_match = re.search(id_pattern, query_lower)
            if id_match:
                parameters["candidate_id"] = id_match.group(1) or id_match.group(2)
            
            # Find time
            time_pattern = r'(?:at|from)\s+(\d+(?::\d+)?(?:\s*[ap]\.?m\.?)?)'
            time_match = re.search(time_pattern, query_lower)
            if time_match:
                parameters["time"] = time_match.group(1)
                
            # Find date
            date_pattern = r'(?:on|for)\s+(tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{2,4}?)'
            date_match = re.search(date_pattern, query_lower)
            if date_match:
                parameters["date"] = date_match.group(1)
            
            return {"query_type": "interview_scheduling", "confidence": 0.9, "parameters": parameters}
    
    # Check for candidate search intent        
    search_terms = ["find", "search", "get", "show", "list", "retrieve", "give me"]
    is_search = any(term in query_lower for term in search_terms)
    
    if is_search or "candidate" in query_lower:
        parameters = {}
        
        # Extract role type
        role_pattern = r'(?:for|with)\s+(?:a|an)?\s*([\w\s]+?)\s*(?:role|position|developer|engineer|designer|specialist)s?'
        role_match = re.search(role_pattern, query_lower)
        if role_match:
            parameters["role_type"] = role_match.group(1).strip()
        
        # Extract count
        count_pattern = r'(\d+|one|two|three|four|five)\s+candidates'
        count_match = re.search(count_pattern, query_lower)
        if count_match:
            count_text = count_match.group(1)
            if count_text.isdigit():
                parameters["required_count"] = int(count_text)
            else:
                # Convert text number to digit
                text_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
                parameters["required_count"] = text_to_num.get(count_text, 3)
        else:
            parameters["required_count"] = 3
        
        # Extract skills
        skills = []
        common_skills = ["python", "java", "javascript", "react", "angular", "node", "aws", "cloud", "fullstack", "frontend", "backend"]
        for skill in common_skills:
            if skill in query_lower:
                skills.append(skill)
        
        if skills:
            parameters["skills"] = skills
        
        return {"query_type": "candidate_search", "confidence": 0.9, "parameters": parameters}
    
    # Default fallback
    return {"query_type": "general_question", "confidence": 0.6, "parameters": {}}
        
  # Enhanced intelligent_interview_matching method for llm_agent.py

  def intelligent_interview_matching(self, candidates, interviewers):
    """
    Advanced interview matching system that optimally pairs candidates with interviewers
    with enhanced category support
    
    Uses a multi-dimensional scoring algorithm that considers:
    1. Skill match between candidate and interviewer expertise
    2. Category match (genai, fullstack, etc.)
    3. Current interviewer workload
    4. Interviewer availability
    5. Diversity of perspectives for candidates
    
    Args:
        candidates: List of candidate objects with skills and experience info
        interviewers: List of interviewer objects with expertise and availability
        
    Returns:
        Dictionary with matched interviews and status information
    """
    # Track scheduling changes
    scheduling_changes = []
    
    # Group candidates by category
    category_groups = {}
    for candidate in candidates:
        # Extract category if available
        candidate_category = None
        if isinstance(candidate, dict) and "category" in candidate:
            candidate_category = candidate["category"]
        elif isinstance(candidate, str):
            category_match = re.search(r'Category:\s+(\w+)', candidate)
            if category_match:
                candidate_category = category_match.group(1)
        
        # Default category if none found
        if not candidate_category:
            candidate_category = "general"
            
        # Add to category group
        if candidate_category not in category_groups:
            category_groups[candidate_category] = []
        
        # Parse string representation if needed
        if isinstance(candidate, str):
            parsed_candidate = self._parse_candidate_from_string(candidate)
            parsed_candidate["category"] = candidate_category
            category_groups[candidate_category].append(parsed_candidate)
        else:
            # Ensure category is included in dict representation
            candidate_dict = candidate.copy() if isinstance(candidate, dict) else {"ID": "unknown", "Name": "Unknown"}
            candidate_dict["category"] = candidate_category
            category_groups[candidate_category].append(candidate_dict)
    
    # Use the LLM as a matching engine with explicit scoring criteria
    system_message = SystemMessage(content="""
      You are an expert AI system for optimal interview matching between candidates and interviewers.
      Your task is to create the most effective pairings based on multiple factors.
      
      For each potential pairing, calculate a composite score (0-100) based on these weighted factors:
      - SKILL MATCH (35%): Alignment between candidate skills and interviewer expertise
      - CATEGORY MATCH (25%): Interviewer's expertise in the candidate's job category
      - WORKLOAD BALANCE (20%): Distribution of interviews among available interviewers
      - AVAILABILITY (10%): Interviewer's available time slots
      - DIVERSITY (10%): Ensuring candidates experience different interviewer perspectives
      
      For each candidate, you must determine:
      1. The best interviewer to conduct the interview based on expertise match
      2. An appropriate interview date (default to tomorrow if not specified)
      3. An available time slot that works for the interviewer
      4. The duration (always 45 minutes)
      
      Your response must be structured as a valid JSON array where each element contains complete details 
      about a single candidate-interviewer pairing.
    """)
    
    # Prepare interviewer data
    formatted_interviewers = []
    for interviewer in interviewers:
        if isinstance(interviewer, dict):
            # Ensure standard format
            if "expertise" not in interviewer and "skills" in interviewer:
                interviewer["expertise"] = interviewer["skills"]
            
            formatted_interviewers.append(interviewer)
    
    # Create category-expertise mapping for interviewers
    interviewer_category_expertise = {}
    for interviewer in formatted_interviewers:
        interviewer_id = interviewer.get("id", "")
        expertise_list = interviewer.get("expertise", [])
        
        # Map expertise to categories
        category_expertise = {}
        for category in category_groups.keys():
            # Calculate a score for this category
            if category.lower() in [exp.lower() for exp in expertise_list]:
                category_expertise[category] = 1.0  # Direct match
            else:
                # Check for related expertise
                related_score = 0.0
                if category.lower() == "genai" and any(exp.lower() in ["ai", "machine learning", "nlp"] for exp in expertise_list):
                    related_score = 0.8
                elif category.lower() == "fullstack" and any(exp.lower() in ["frontend", "backend", "web"] for exp in expertise_list):
                    related_score = 0.8
                elif category.lower() == "frontend" and any(exp.lower() in ["ui", "ux", "web", "javascript"] for exp in expertise_list):
                    related_score = 0.8
                elif category.lower() == "backend" and any(exp.lower() in ["server", "api", "database"] for exp in expertise_list):
                    related_score = 0.8
                else:
                    related_score = 0.5  # Default partial match
                
                category_expertise[category] = related_score
        
        interviewer_category_expertise[interviewer_id] = category_expertise
    
    # Create the consolidated prompt with category information
    candidates_with_categories = []
    for category, candidate_list in category_groups.items():
        for candidate in candidate_list:
            # Add category to the candidate dict
            candidate_copy = candidate.copy() if isinstance(candidate, dict) else {}
            candidate_copy["category"] = category
            candidates_with_categories.append(candidate_copy)
    
    user_message = HumanMessage(content=f"""
      CANDIDATES TO SCHEDULE:
      {json.dumps(candidates_with_categories, indent=2)}
      
      AVAILABLE INTERVIEWERS:
      {json.dumps(formatted_interviewers, indent=2)}
      
      CATEGORY EXPERTISE MAPPING:
      {json.dumps(interviewer_category_expertise, indent=2)}
      
      CONSTRAINTS:
      - Each interview should last 45 minutes
      - Schedule within the next 5 business days
      - Respect interviewer availability
      - Balance workload across interviewers
      - Prioritize skill matching
      - Match candidates with interviewers experienced in their category
      
      For each candidate, determine the optimal interviewer and time slot.
      
      Return your result as a JSON array with this exact structure:
      [
        {{
          "candidate_id": "id_string",
          "candidate_name": "Full Name",
          "candidate_key_skills": ["skill1", "skill2"],
          "candidate_category": "category_name",
          "interviewer_id": "interviewer_id_string",
          "interviewer_name": "Interviewer Name",
          "interviewer_expertise": ["expertise1", "expertise2"],
          "expertise_match_score": 85.5,
          "category_match_score": 90.0,
          "interview_date": "YYYY-MM-DD",
          "start_time": "HH:MM",
          "end_time": "HH:MM",
          "meeting_link": "https://meet.google.com/xxx-yyyy-zzz"
        }},
        // more interviews...
      ]
      
      IMPORTANT: Return ONLY the JSON array - no explanations, no comments, just the pure JSON array.
    """)
    
    try:
        # First try using the LLM for intelligently matching interviews
        response = self.llm.invoke([system_message, user_message])
        
        # Extract JSON from the response
        matches = None
        try:
            # Find JSON array pattern in the response
            json_pattern = r'\[\s*\{.*\}\s*\]'
            json_match = re.search(json_pattern, response.content, re.DOTALL)
            
            if json_match:
                matches = json.loads(json_match.group(0))
            else:
                # Try loading the entire response as JSON
                matches = json.loads(response.content)
                
            # Validate the matches structure
            if not isinstance(matches, list) or not matches:
                raise ValueError("Invalid matches format: not a list or empty list")
                
            # Ensure each match has required fields
            required_fields = ["candidate_id", "candidate_name", "interviewer_name", "start_time"]
            for match in matches:
                if not all(field in match for field in required_fields):
                    raise ValueError(f"Missing required fields in match: {match}")
                    
            # Generate meeting links if missing
            for match in matches:
                if "meeting_link" not in match or not match["meeting_link"]:
                    match["meeting_link"] = f"https://meet.google.com/{self._generate_meeting_id()}"
                    
            # Add default end time if missing (45 minutes from start time)
            for match in matches:
                if "end_time" not in match or not match["end_time"]:
                    start_time = match.get("start_time", "09:00")
                    match["end_time"] = self._calculate_end_time(start_time, 45)
                    
            # Calculate scheduling fairness
            self.evaluate_scheduling_fairness(matches)
                    
            return {
                "success": True,
                "matches": matches,
                "changes": scheduling_changes
            }
                
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, fall back to fallback matching
            print(f"Error parsing LLM matching response: {e}")
            print(f"Raw response: {response.content}")
            matches = None
    except Exception as e:
        print(f"Error in intelligent interview matching: {str(e)}")
        matches = None
    
    # If LLM matching failed, fall back to rule-based matching
    if not matches:
        print("Using fallback matching algorithm")
        manual_matches = self._fallback_matching_algorithm(candidates_with_categories, formatted_interviewers)
        scheduling_changes.append({"message": "Using fallback scheduling algorithm due to matching issues"})
        
        # Calculate scheduling fairness metrics for the fallback matches
        self.evaluate_scheduling_fairness(manual_matches)
        
        return {
            "success": True,
            "matches": manual_matches,
            "changes": scheduling_changes
        }

  def _generate_meeting_id(self):
    """Generate a random Google Meet ID"""
    chars = "abcdefghijklmnopqrstuvwxyz"
    nums = "0123456789"
    
    part1 = ''.join(random.choice(chars) for _ in range(3))
    part2 = ''.join(random.choice(chars + nums) for _ in range(4))
    part3 = ''.join(random.choice(chars) for _ in range(3))
    
    return f"{part1}-{part2}-{part3}"

  def _calculate_end_time(self, start_time, duration_minutes):
    """Calculate end time given start time and duration"""
    try:
        # Parse start time
        if ":" in start_time:
            hours, minutes = map(int, start_time.split(":"))
        else:
            # Handle format like "9am"
            is_pm = "pm" in start_time.lower()
            time_value = start_time.lower().replace("am", "").replace("pm", "").strip()
            hours = int(time_value)
            minutes = 0
            
            if is_pm and hours < 12:
                hours += 12
        
        # Calculate end time
        total_minutes = hours * 60 + minutes + duration_minutes
        end_hours = total_minutes // 60
        end_minutes = total_minutes % 60
        
        # Format as HH:MM
        return f"{end_hours:02d}:{end_minutes:02d}"
    except Exception as e:
        print(f"Error calculating end time: {e}")
        # Default to 45 minutes later
        return "10:00" if start_time == "09:15" else "10:15"

  # Improved parse_candidate_from_string method

  def _parse_candidate_from_string(self, candidate_str):
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
    
    # Extract name with enhanced patterns
    # First look for explicit name fields
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
    
    # If no explicit name, try to extract from the beginning of the resume
    # which often contains name, email, phone in that order
    if "Name" not in candidate_data:
        # Try to find name at the beginning of the resume
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
    
    # If no explicit skills section, extract skills by keywords
    if not skills_found:
        # Common technical skills to look for
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
    
    # If still no skills found, use generic skill
    if not skills_found or "Skills" not in candidate_data:
        candidate_data["Skills"] = ["General"]
    
    # Store the original resume text for reference
    candidate_data["Resume"] = candidate_str
    
    return candidate_data

  def schedule_interview_for_candidate(self, scheduler, request_data, resume_list):
    """
    Enhanced method to schedule an interview for a specific candidate with category support
    
    Features:
    - Better candidate identification
    - Category-based interviewer matching
    - Support for specific interviewer requests
    - Intelligent time selection when requested time is unavailable
    """
    try:
        # Extract information from the request with enhanced parsing
        candidate_id = request_data.get("candidate_id")
        interview_time = request_data.get("time", "9am")
        interview_date = request_data.get("date", "tomorrow")
        requested_interviewer = request_data.get("interviewer")
        
        # Find the candidate in the resume list with semantic matching
        target_candidate = None
        candidate_selected_note = None
        
        if candidate_id:
            # Try to find by exact ID first
            for candidate in resume_list:
                if isinstance(candidate, dict) and str(candidate.get("ID", "")) == str(candidate_id):
                    target_candidate = candidate
                    break
                elif isinstance(candidate, str):
                    # Try different ID patterns in the string
                    id_patterns = [
                        f"ID: {candidate_id}",
                        f"ID:{candidate_id}",
                        f"Applicant ID {candidate_id}",
                        f"Applicant ID: {candidate_id}"
                    ]
                    if any(pattern in candidate for pattern in id_patterns):
                        # Parse string representation
                        target_candidate = self._parse_candidate_from_string(candidate)
                        break
        
        # If candidate not found by ID, use semantic similarity to find the best match
        # or default to the first one if only one candidate exists
        if not target_candidate and resume_list:
            if len(resume_list) == 1:
                # Only one candidate, use that one
                target_candidate = resume_list[0]
                if isinstance(target_candidate, str):
                    target_candidate = self._parse_candidate_from_string(target_candidate)
            else:
                # Multiple candidates, use the first one and warn in the response
                target_candidate = resume_list[0]
                if isinstance(target_candidate, str):
                    target_candidate = self._parse_candidate_from_string(target_candidate)
                
                # Add a note about using the first candidate
                candidate_selected_note = "No specific candidate ID was provided, so I've scheduled for the first candidate in our results."
        
        if not target_candidate:
            return {
                "success": False,
                "message": "Could not identify the candidate for scheduling. Please search for candidates first or provide a valid candidate ID."
            }
        
        # Ensure we have an ID
        if not target_candidate.get("ID") and isinstance(target_candidate, dict):
            # Generate a random ID if none exists
            target_candidate["ID"] = str(random.randint(1000, 9999))
        
        # Extract candidate skills for better interviewer matching
        candidate_skills = target_candidate.get("Skills", []) if isinstance(target_candidate, dict) else []
        if not candidate_skills and isinstance(target_candidate, dict) and "Resume" in target_candidate:
            # Try to extract skills from resume text
            skills = self._extract_skills_from_text(target_candidate["Resume"])
            if skills:
                candidate_skills = skills
                target_candidate["Skills"] = skills
        
        # Extract candidate category if available
        candidate_category = None
        if isinstance(target_candidate, dict) and "category" in target_candidate:
            candidate_category = target_candidate["category"]
        elif isinstance(target_candidate, str):
            # Try to extract category from string
            category_match = re.search(r'Category:\s+(\w+)', target_candidate)
            if category_match:
                candidate_category = category_match.group(1)
                
        # Use skills and category to find the best interviewer match if no specific interviewer requested
        best_match_params = {
            "candidate_skills": candidate_skills,
            "preferred_interviewer": requested_interviewer,
            "interview_time": interview_time,
            "interview_date": interview_date,
            "category": candidate_category
        }
        
        # Schedule the interview with enhanced matchmaking
        scheduled_interview = scheduler.schedule_interview_with_best_match(
            target_candidate, 
            best_match_params
        )
        
        if not scheduled_interview:
            return {
                "success": False,
                "message": "Could not schedule the interview. All interviewers might be unavailable at the requested time."
            }
        
        # If we have a category, add it to the interview data
        if candidate_category and isinstance(scheduled_interview, dict):
            scheduled_interview["category"] = candidate_category
        
        # Generate a draft invitation with personalized content
        invitation_draft = scheduler.generate_invitation_draft(scheduled_interview)
        
        # Prepare success message
        success_message = f"Successfully scheduled an interview for candidate {target_candidate.get('Name', 'Unknown')} with {scheduled_interview.get('interviewer_name', 'Unknown')}."
        
        # Add note about candidate selection if applicable
        if candidate_selected_note:
            success_message = f"{candidate_selected_note} {success_message}"
        
        return {
            "success": True,
            "interview": scheduled_interview,
            "invitation_draft": invitation_draft,
            "message": success_message
        }
    except Exception as e:
        import traceback
        print(f"Error scheduling interview: {str(e)}")
        print(traceback.format_exc())
        return {
            "success": False,
            "message": f"Failed to schedule interview: {str(e)}"
        }

        
  def _extract_skills_from_text(self, text):
    """Helper method to extract skills from resume text"""
    if not text:
        return []
        
    # Common technical skills to look for
    skill_keywords = [
        "Python", "JavaScript", "Java", "C#", "C++", "Go", "Ruby", "PHP",
        "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "HTML", "CSS",
        "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Redis",
        "DevOps", "CI/CD", "Git", "Jenkins", "Agile", "Scrum",
        "ML", "AI", "TensorFlow", "PyTorch", "NLP", "Computer Vision",
        "Mobile", "Android", "iOS", "Swift", "Kotlin",
        "Fullstack", "Frontend", "Backend", "Data Science", "Cloud", "GenAI"
    ]
    
    found_skills = []
    for skill in skill_keywords:
        # Look for the skill as a whole word
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)
    
    return found_skills

  def _extract_matches_from_text(self, text, candidates, interviewers):
    """Helper method to extract structured matching data from LLM text output"""
    matches = []
    
    # Simple pattern matching to extract candidate-interviewer pairs
    candidate_ids = [str(c.get("ID", "")) for c in candidates]
    interviewer_names = [i.get("name", "") for i in interviewers]
    
    for candidate in candidates:
        candidate_id = str(candidate.get("ID", ""))
        candidate_name = candidate.get("Name", "")
        
        # Find mentions of this candidate in the text
        candidate_pattern = f"(?:Candidate|candidate|Applicant|applicant)\\s*(?:#?\\s*{candidate_id}|{candidate_name})"
        candidate_matches = re.finditer(candidate_pattern, text)
        
        for match in candidate_matches:
            # Look for interviewer assignments in the following text
            start_pos = match.end()
            next_100_chars = text[start_pos:start_pos + 200]
            
            # Find the first interviewer mentioned
            assigned_interviewer = None
            for interviewer in interviewers:
                interviewer_name = interviewer.get("name", "")
                if interviewer_name in next_100_chars:
                    assigned_interviewer = interviewer
                    break
            
            if assigned_interviewer:
                # Look for date/time information
                date_match = re.search(r'(\d{1,2}(?:\/|-)\d{1,2}(?:\/|-)\d{2,4}|\w+ \d{1,2}(?:st|nd|rd|th)?|\w+day)', next_100_chars)
                time_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))', next_100_chars)
                
                interview_date = date_match.group(1) if date_match else "tomorrow"
                interview_time = time_match.group(1) if time_match else "9:00 AM"
                
                # Create a match entry
                match_entry = {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "interviewer_id": assigned_interviewer.get("id", ""),
                    "interviewer_name": assigned_interviewer.get("name", ""),
                    "interview_date": interview_date,
                    "interview_time": interview_time,
                    "duration": "45 minutes"
                }
                
                matches.append(match_entry)
                break  # Only use the first match for each candidate
    
    return matches

  # Enhanced generate_personalized_invitations method for llm_agent.py

  def generate_personalized_invitations(self, matched_interviews):
    """
    Generate personalized and engaging interview invitation drafts for candidates
    with enhanced category awareness
    
    Args:
        matched_interviews: List of interviews with candidate and interviewer details
        
    Returns:
        Dictionary mapping candidate IDs to personalized invitation texts
    """
    system_message = SystemMessage(content="""
      You are an expert in talent acquisition communications with excellent writing skills.
      Create personalized interview invitation emails that are:
      
      1. PROFESSIONAL: Maintain a professional but warm tone throughout
      2. PERSONALIZED: Reference the candidate's specific background and skills
      3. CATEGORY-SPECIFIC: Tailor the content to the candidate's job category
      4. INFORMATIVE: Include all essential details about the interview
      5. CLEAR: Provide explicit instructions for next steps
      6. ENGAGING: Generate excitement about the opportunity
      
      Each email should feel like it was written specifically for that candidate.
      
      IMPORTANT: Create a separate, complete email for EACH candidate in the list.
      If a category is specified, customize the content to that specific job category.
    """)
    
    # Add category information to the prompt
    categories_info = {}
    for interview in matched_interviews:
        category = interview.get("category", None)
        if category:
            if category not in categories_info:
                categories_info[category] = []
            categories_info[category].append(interview.get("candidate_id", ""))
    
    category_prompt = ""
    if categories_info:
        category_prompt = "\n\nCandidate categories:\n"
        for category, candidate_ids in categories_info.items():
            id_list = ", ".join(candidate_ids)
            category_prompt += f"- {category.upper()}: {id_list}\n"
    
    user_message = HumanMessage(content=f"""
      Please generate personalized interview invitation drafts for these matched interviews:
      {matched_interviews}
      {category_prompt}
      
      For each invitation, include:
      1. Personalized greeting using candidate's name
      2. Position-specific introduction referencing the candidate's job category if specified
      3. Complete interview details (date, time, format, interviewer name)
      4. Preparation instructions including topics relevant to their category
      5. Technical setup instructions for the video conference
      6. Clear next steps and contact information for questions
      7. Professional closing
      
      The company values innovation, collaboration, and technical excellence.
      Convey enthusiasm about the candidate's potential fit with our team.
      
      Format each email with "CANDIDATE_ID: [id]" at the beginning.
    """)
    
    try:
        response = self.llm.invoke([system_message, user_message])
        
        # Process the response into individual invitations
        invitation_dict = {}
        
        # Try to detect if the response is in JSON format first
        try:
            json_response = json.loads(response.content)
            if isinstance(json_response, list):
                for invitation in json_response:
                    candidate_id = invitation.get("candidate_id", "")
                    invitation_text = invitation.get("email_draft", "")
                    invitation_dict[candidate_id] = invitation_text
                return invitation_dict
        except json.JSONDecodeError:
            pass
        
        # Try to detect if the response contains multiple invitations
        email_pattern = r"CANDIDATE_ID:\s*(\w+)(.*?)(?=CANDIDATE_ID:|$)"
        email_matches = re.findall(email_pattern, response.content, re.DOTALL)
        
        if email_matches:
            # Extract candidate IDs and content from each match
            for candidate_id, content in email_matches:
                invitation_dict[candidate_id.strip()] = content.strip()
        else:
            # Alternative: Try to extract by Subject line
            subject_pattern = r"Subject:.*?(?=Subject:|$)"
            email_drafts = re.findall(subject_pattern, response.content, re.DOTALL)
            
            if email_drafts and len(email_drafts) == len(matched_interviews):
                # Match emails to interviews in order
                for i, interview in enumerate(matched_interviews):
                    candidate_id = interview.get("candidate_id", "")
                    if candidate_id and i < len(email_drafts):
                        invitation_dict[candidate_id] = email_drafts[i].strip()
            else:
                # Try looking for candidate names in the drafts
                for interview in matched_interviews:
                    candidate_id = interview.get("candidate_id", "")
                    candidate_name = interview.get("candidate_name", "")
                    
                    if candidate_id and candidate_name:
                        # Look for greeting with this candidate's name
                        name_pattern = rf"Dear\s+{re.escape(candidate_name)}"
                        match = re.search(name_pattern, response.content)
                        
                        if match:
                            # Extract from this match to the next greeting or end
                            start_pos = match.start()
                            next_match = re.search(r"Dear\s+", response.content[start_pos+len(match.group(0)):])
                            
                            if next_match:
                                end_pos = start_pos + len(match.group(0)) + next_match.start()
                                invitation_dict[candidate_id] = response.content[start_pos:end_pos].strip()
                            else:
                                # No more greetings, use until end
                                invitation_dict[candidate_id] = response.content[start_pos:].strip()
        
        # If still no invitations, create fallback versions
        if not invitation_dict:
            for interview in matched_interviews:
                candidate_id = interview.get("candidate_id", "")
                category = interview.get("category", "")
                
                if candidate_id:
                    invitation_dict[candidate_id] = self._generate_category_based_invitation(
                        interview.get("candidate_name", "Candidate"),
                        interview.get("interview_date", "tomorrow"),
                        interview.get("start_time", "9:00 AM"),
                        interview.get("end_time", "9:45 AM"),
                        interview.get("interviewer_name", "Interviewer"),
                        interview.get("meeting_link", "https://meet.google.com/"),
                        category
                    )
        
        return invitation_dict
    except Exception as e:
        print(f"Error generating interview invitations: {e}")
        # Create fallback invitations
        invitation_dict = {}
        for interview in matched_interviews:
            candidate_id = interview.get("candidate_id", "")
            category = interview.get("category", "")
            
            if candidate_id:
                invitation_dict[candidate_id] = self._generate_category_based_invitation(
                    interview.get("candidate_name", "Candidate"),
                    interview.get("interview_date", "tomorrow"),
                    interview.get("start_time", "9:00 AM"),
                    interview.get("end_time", "9:45 AM"),
                    interview.get("interviewer_name", "Interviewer"),
                    interview.get("meeting_link", "https://meet.google.com/"),
                    category
                )
        return invitation_dict

    def _generate_category_based_invitation(self, candidate_name, date, start_time, end_time, interviewer_name, meeting_link, category=""):
      """
      Generate a category-specific invitation based on the candidate's job category
      
      Args:
          candidate_ID: ID of the candidate
          date: Interview date
          start_time: Interview start time
          end_time: Interview end time
          interviewer_name: Name of the interviewer
          meeting_link: Meeting link for the interview
          category: Job category (genai, fullstack, frontend, backend, etc.)
          
      Returns:
          Formatted invitation text
      """
      # Set job role and preparation topics based on category
      role = "Software Engineer"
      preparation_topics = "general software development, coding practice, and problem-solving"
      
      if category.lower() == "genai":
          role = "GenAI Engineer"
          preparation_topics = "generative AI concepts, large language models, prompt engineering, and AI application development"
      elif category.lower() == "fullstack":
          role = "Full Stack Developer"
          preparation_topics = "both frontend and backend technologies, system architecture, and full-stack development workflows"
      elif category.lower() == "frontend":
          role = "Frontend Developer"
          preparation_topics = "UI/UX design, JavaScript frameworks, responsive design, and frontend development best practices"
      elif category.lower() == "backend":
          role = "Backend Developer"
          preparation_topics = "server-side technologies, databases, API development, and system architecture"
      
      invitation = f"""
  Subject: Interview Invitation

  Dear {candidate_name},

  We're excited to invite you to an interview at our company. After reviewing your impressive qualifications and experience, we believe you would be a valuable addition to our team.

  Interview Details:
  - Date: {date}
  - Time: {start_time} - {end_time}
  - Format: Video Conference
  - Meeting Link: {meeting_link}
  - Interviewer: {interviewer_name}

  During the interview, we'll explore your experience with the domain topics. We're particularly interested in discussing your approach to problem-solving and your previous projects.

  Please ensure you have a stable internet connection, a working webcam, and a quiet environment for the interview.

  If you need to reschedule or have any questions, please reply to this email or contact our HR department at hr@company.com.

  We look forward to speaking with you!

  Best regards,
  Recruiting Team
  Siemens
  """
      
      return invitation.strip()

  def _fallback_matching_algorithm(self, candidates, interviewers):
    """Enhanced fallback matching with category support"""
    matches = []
    interviewer_counts = {interviewer.get("id", i): 0 for i, interviewer in enumerate(interviewers)}
    
    # Keep track of scheduled slots to avoid conflicts
    scheduled_slots = {}  # Format: {interviewer_id: [(start_hour, start_minute, end_hour, end_minute)]}
    
    for candidate in candidates:
        candidate_id = candidate.get("ID", "")
        candidate_name = candidate.get("Name", "")
        candidate_skills = candidate.get("Skills", [])
        candidate_category = candidate.get("category", "")
        
        # Find the best interviewer based on skills, category and current load
        best_interviewer = None
        best_score = -1
        best_time_slot = None
        
        for interviewer in interviewers:
            interviewer_id = interviewer.get("id", "")
            interviewer_name = interviewer.get("name", "")
            interviewer_skills = interviewer.get("expertise", [])
            interviewer_availability = interviewer.get("availability", list(range(9, 17)))  # Default 9am-5pm
            
            # Calculate skill match
            matching_skills = set(candidate_skills).intersection(set(interviewer_skills))
            skill_score = len(matching_skills) * 10
            
            # Calculate category match - higher if interviewer has expertise in this category
            category_score = 0
            if candidate_category and candidate_category.lower() in [s.lower() for s in interviewer_skills]:
                category_score = 50  # Direct category match
            elif candidate_category == "genai" and any(s.lower() in ["ai", "machine learning"] for s in interviewer_skills):
                category_score = 40  # Related skills for GenAI
            elif candidate_category == "fullstack" and any(s.lower() in ["frontend", "backend"] for s in interviewer_skills):
                category_score = 40  # Related skills for fullstack
            else:
                category_score = 20  # Default partial match
            
            # Factor in current workload (lower is better)
            workload_factor = 50 - (interviewer_counts.get(interviewer_id, 0) * 10)  # Decreases as workload increases
            
            # Get already scheduled slots for this interviewer
            interviewer_slots = scheduled_slots.get(interviewer_id, [])
            
            # Find available time slots
            for hour in sorted(interviewer_availability):
                for minute in [0, 15, 30, 45]:
                    # Check if this slot works
                    start_time = (hour, minute)
                    end_hour = hour + ((minute + 45) // 60)
                    end_minute = (minute + 45) % 60
                    end_time = (end_hour, end_minute)
                    
                    # Check if slot conflicts with already scheduled interviews
                    conflict = False
                    for slot_start, slot_end in interviewer_slots:
                        # Convert to decimal hours for easier comparison
                        slot_start_decimal = slot_start[0] + (slot_start[1] / 60)
                        slot_end_decimal = slot_end[0] + (slot_end[1] / 60)
                        start_decimal = start_time[0] + (start_time[1] / 60)
                        end_decimal = end_time[0] + (end_time[1] / 60)
                        
                        # Check for overlap
                        if not (end_decimal <= slot_start_decimal or start_decimal >= slot_end_decimal):
                            conflict = True
                            break
                    
                    if not conflict:
                        # This slot works - calculate the total score
                        total_score = skill_score + category_score + workload_factor
                        
                        # Time preference score - earlier is slightly better
                        time_score = 20 - (hour - 9)
                        total_score += time_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_interviewer = interviewer
                            best_time_slot = (start_time, end_time)
        
        # If we found a match, use it
        if best_interviewer and best_time_slot:
            # Update interviewer workload count
            interviewer_id = best_interviewer.get("id", "")
            interviewer_counts[interviewer_id] = interviewer_counts.get(interviewer_id, 0) + 1
            
            # Update scheduled slots
            if interviewer_id not in scheduled_slots:
                scheduled_slots[interviewer_id] = []
            scheduled_slots[interviewer_id].append(best_time_slot)
            
            # Format time strings
            start_hour, start_minute = best_time_slot[0]
            end_hour, end_minute = best_time_slot[1]
            
            start_time_str = f"{start_hour:02d}:{start_minute:02d}"
            end_time_str = f"{end_hour:02d}:{end_minute:02d}"
            
            # Get interviewer expertise
            interviewer_expertise = best_interviewer.get("expertise", [])
            
            # Calculate expertise match score
            matching_skills = set(candidate_skills).intersection(set(interviewer_expertise))
            expertise_match_score = len(matching_skills) / max(len(candidate_skills), 1) * 100
            
            # Calculate category match score
            category_match_score = 0
            if candidate_category and candidate_category.lower() in [s.lower() for s in interviewer_expertise]:
                category_match_score = 100  # Direct category match
            elif candidate_category == "genai" and any(s.lower() in ["ai", "machine learning"] for s in interviewer_expertise):
                category_match_score = 80  # Related skills for GenAI
            elif candidate_category == "fullstack" and any(s.lower() in ["frontend", "backend"] for s in interviewer_expertise):
                category_match_score = 80  # Related skills for fullstack
            else:
                category_match_score = 50  # Default partial match
            
            # Create match entry
            match_entry = {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "candidate_key_skills": candidate_skills,
                "candidate_category": candidate_category,
                "interviewer_id": interviewer_id,
                "interviewer_name": best_interviewer.get("name", ""),
                "interviewer_expertise": interviewer_expertise,
                "expertise_match_score": expertise_match_score,
                "category_match_score": category_match_score,
                "interview_date": "tomorrow",  # Default to tomorrow
                "start_time": start_time_str,
                "end_time": end_time_str,
                "meeting_link": f"https://meet.google.com/{self._generate_meeting_id()}"
            }
            
            matches.append(match_entry)
        else:
            # Extreme fallback - assign to first interviewer with default time
            fallback_interviewer = interviewers[0] if interviewers else {"id": "default", "name": "Default Interviewer", "expertise": []}
            interviewer_id = fallback_interviewer.get("id", "default")
            
            # Find any available time slot
            available_hour = 9
            available_minute = 0
            
            # Try time slots until we find one that's open
            for hour in range(9, 18):
                for minute in [0, 15, 30, 45]:
                    slot_conflict = False
                    interviewer_slots = scheduled_slots.get(interviewer_id, [])
                    
                    for slot_start, slot_end in interviewer_slots:
                        slot_start_decimal = slot_start[0] + (slot_start[1] / 60)
                        slot_end_decimal = slot_end[0] + (slot_end[1] / 60)
                        start_decimal = hour + (minute / 60)
                        end_decimal = hour + ((minute + 45) / 60)
                        
                        if not (end_decimal <= slot_start_decimal or start_decimal >= slot_end_decimal):
                            slot_conflict = True
                            break
                    
                    if not slot_conflict:
                        available_hour = hour
                        available_minute = minute
                        break
                
                if available_hour != 9 or available_minute != 0:
                    break
            
            # Calculate end time
            end_hour = available_hour + ((available_minute + 45) // 60)
            end_minute = (available_minute + 45) % 60
            
            # Format time strings
            start_time_str = f"{available_hour:02d}:{available_minute:02d}"
            end_time_str = f"{end_hour:02d}:{end_minute:02d}"
            
            # Update scheduled slots
            if interviewer_id not in scheduled_slots:
                scheduled_slots[interviewer_id] = []
            scheduled_slots[interviewer_id].append(((available_hour, available_minute), (end_hour, end_minute)))
            
            # Create match entry
            match_entry = {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "candidate_key_skills": candidate_skills,
                "candidate_category": candidate_category,
                "interviewer_id": interviewer_id,
                "interviewer_name": fallback_interviewer.get("name", "Default Interviewer"),
                "interviewer_expertise": fallback_interviewer.get("expertise", []),
                "expertise_match_score": 0,  # No expertise match in fallback
                "category_match_score": 0,  # No category match in fallback
                "interview_date": "tomorrow",  # Default to tomorrow
                "start_time": start_time_str,
                "end_time": end_time_str,
                "meeting_link": f"https://meet.google.com/{self._generate_meeting_id()}"
            }
            
            matches.append(match_entry)
    
    return matches

  def optimize_interviewer_workload(self, interviewer_assignments):
    """
    Optimizes and rebalances interviewer workloads for fairer distribution
    
    Features:
    - Algorithmic workload balancing
    - Preservation of skill matching where possible
    - Handling of interviewer availability constraints
    - Support for different prioritization strategies
    
    Args:
        interviewer_assignments: Current interview assignments
        
    Returns:
        Rebalanced assignment list
    """
    # Extract current workload distribution
    interviewer_loads = {}
    for assignment in interviewer_assignments:
        interviewer_id = assignment.get("interviewer_id", "")
        if interviewer_id not in interviewer_loads:
            interviewer_loads[interviewer_id] = []
        interviewer_loads[interviewer_id].append(assignment)
    
    # Calculate current imbalance metrics
    loads = [len(interviews) for interviews in interviewer_loads.values()]
    max_load = max(loads) if loads else 0
    min_load = min(loads) if loads else 0
    imbalance = max_load - min_load
    
    # If already balanced, return original assignments
    if imbalance <= 1:
        return interviewer_assignments
    
    # Use the LLM to suggest a rebalanced schedule
    system_message = SystemMessage(content="""
      You are an expert AI system for optimizing interviewer workload distribution.
      Your goal is to create a fair, balanced distribution of interviews while maintaining:
      1. Strong skill matching between candidates and interviewers
      2. Reasonable scheduling (no back-to-back interviews)
      3. Respect for availability constraints
      
      Optimize for both fairness (equal number of interviews per interviewer) and quality (good skill matches).
    """)
    
    # Get information about interviewers for rebalancing
    interviewer_details = []
    for interviewer_id, assignments in interviewer_loads.items():
        # Extract interviewer info from the first assignment
        first_assignment = assignments[0]
        interviewer_name = first_assignment.get("interviewer_name", "")
        
        interviewer_details.append({
            "id": interviewer_id,
            "name": interviewer_name,
            "current_load": len(assignments),
            "current_assignments": [
                {
                    "candidate_id": a.get("candidate_id", ""),
                    "candidate_name": a.get("candidate_name", ""),
                    "date": a.get("interview_date", ""),
                    "time": a.get("interview_time", "")
                } for a in assignments
            ]
        })
    
    user_message = HumanMessage(content=f"""
      CURRENT INTERVIEWER WORKLOAD:
      {json.dumps(interviewer_details, indent=2)}
      
      OPTIMIZATION GOALS:
      1. Balance the number of interviews per interviewer (ideally within ±1)
      2. Maintain appropriate skill matching where possible
      3. Avoid scheduling conflicts
      4. Minimize changes to the current schedule
      
      Suggest a rebalanced schedule by:
      1. Identifying which interviews should be reassigned
      2. Specifying which interviewer should take each reassigned interview
      3. Explaining the reasoning for each change
      
      Return the optimized schedule as a JSON array.
    """)
    
    try:
        response = self.llm.invoke([system_message, user_message])
        # Try to parse JSON from the response
        try:
            optimized_assignments = json.loads(response.content)
            return optimized_assignments
        except:
            # If JSON parsing fails, use regex to extract the suggestions
            return self._process_workload_optimization_text(response.content, interviewer_assignments)
    except Exception as e:
        print(f"Error in workload optimization: {str(e)}")
        # Apply a simple rule-based rebalancing
        return self._simple_workload_rebalancing(interviewer_loads, interviewer_assignments)

  def _process_workload_optimization_text(self, text, original_assignments):
    """Extract optimization suggestions from text response"""
    # Extract any reassignment suggestions using regex
    reassignment_pattern = r"(?:Move|Reassign|Transfer)\s+(?:interview\s+(?:for|with))?\s+(?:candidate\s+)?([^(]+?)\s*(?:\(ID: (\d+)\))?\s+from\s+([^(]+?)\s+to\s+([^(]+?)\s+(?=\.|$)"
    reassignments = re.findall(reassignment_pattern, text, re.IGNORECASE)
    
    # Create a copy of the original assignments to modify
    optimized = original_assignments.copy()
    
    for match in reassignments:
        candidate_name, candidate_id, from_interviewer, to_interviewer = match
        
        # Clean up extracted text
        candidate_name = candidate_name.strip()
        from_interviewer = from_interviewer.strip()
        to_interviewer = to_interviewer.strip()
        
        # Find the assignment to change
        for i, assignment in enumerate(optimized):
            assignment_candidate_name = assignment.get("candidate_name", "").strip()
            assignment_candidate_id = assignment.get("candidate_id", "")
            assignment_interviewer = assignment.get("interviewer_name", "").strip()
            
            # Check if this is the assignment to modify
            if (assignment_candidate_name.lower() == candidate_name.lower() or 
                (candidate_id and assignment_candidate_id == candidate_id)) and \
                assignment_interviewer.lower() == from_interviewer.lower():
                
                # Find the target interviewer ID
                for other_assignment in original_assignments:
                    other_interviewer = other_assignment.get("interviewer_name", "").strip()
                    if other_interviewer.lower() == to_interviewer.lower():
                        # Update the assignment
                        optimized[i]["interviewer_name"] = other_interviewer
                        optimized[i]["interviewer_id"] = other_assignment.get("interviewer_id", "")
                        break
                
                break
    
    return optimized

  def _simple_workload_rebalancing(self, interviewer_loads, original_assignments):
    """Simple rule-based workload balancing algorithm"""
    # Find overloaded and underloaded interviewers
    avg_load = sum(len(assignments) for assignments in interviewer_loads.values()) / len(interviewer_loads)
    overloaded = []
    underloaded = []
    
    for interviewer_id, assignments in interviewer_loads.items():
        if len(assignments) > avg_load + 0.5:
            overloaded.append((interviewer_id, assignments))
        elif len(assignments) < avg_load - 0.5:
            underloaded.append((interviewer_id, assignments))
    
    # Sort overloaded by most interviews, underloaded by fewest
    overloaded.sort(key=lambda x: len(x[1]), reverse=True)
    underloaded.sort(key=lambda x: len(x[1]))
    
    # Create a copy of the original assignments to modify
    optimized = original_assignments.copy()
    
    # Redistribute interviews
    while overloaded and underloaded:
        over_id, over_assignments = overloaded[0]
        under_id, under_assignments = underloaded[0]
        
        # Move one interview from overloaded to underloaded
        assignment_to_move = over_assignments.pop()
        
        # Find this assignment in the optimized list
        for i, assignment in enumerate(optimized):
            if (assignment.get("interviewer_id", "") == over_id and
                assignment.get("candidate_id", "") == assignment_to_move.get("candidate_id", "")):
                
                # Update the interviewer information
                target_interviewer_name = None
                for orig_assignment in original_assignments:
                    if orig_assignment.get("interviewer_id", "") == under_id:
                        target_interviewer_name = orig_assignment.get("interviewer_name", "")
                        break
                
                if target_interviewer_name:
                    optimized[i]["interviewer_id"] = under_id
                    optimized[i]["interviewer_name"] = target_interviewer_name
                
                # Update the tracking lists
                under_assignments.append(assignment_to_move)
                break
        
        # Recalculate and possibly remove from lists
        if len(over_assignments) <= avg_load + 0.5:
            overloaded.pop(0)
        else:
            # Re-sort overloaded
            overloaded.sort(key=lambda x: len(x[1]), reverse=True)
        
        if len(under_assignments) >= avg_load - 0.5:
            underloaded.pop(0)
        else:
            # Re-sort underloaded
            underloaded.sort(key=lambda x: len(x[1]))
    
    # Calculate scheduling fairness metrics for the optimized schedule
    self.evaluate_scheduling_fairness(optimized)
    
    return optimized

  def evaluate_shortlisting_accuracy(self, shortlisted_candidates, job_description=None):
    """
    Evaluate the accuracy of resume shortlisting
    
    Args:
        shortlisted_candidates: Dictionary or list of shortlisted candidates
        job_description: Optional job description for comparison
        
    Returns:
        Dictionary with accuracy metrics
    """
    metrics = {
        "shortlisting_precision": 0,
        "skill_coverage": 0,
        "experience_match": 0,
        "education_match": 0,
        "overall_accuracy": 0
    }
    
    # If we have no candidates or job description, return zeroes
    if not shortlisted_candidates:
        return metrics
        
    # Normalize shortlisted_candidates to a list
    candidates_list = []
    if isinstance(shortlisted_candidates, dict):
        for candidate_id, candidate_info in shortlisted_candidates.items():
            candidate_info["id"] = candidate_id
            candidates_list.append(candidate_info)
    else:
        candidates_list = shortlisted_candidates
        
    # If we have job description, extract key requirements
    required_skills = []
    preferred_skills = []
    min_experience = 0
    required_education = "bachelor's"
    
    if job_description:
        # Extract job requirements using simple pattern matching
        # In a real system, this would use more sophisticated NLP
        skill_keywords = ["python", "java", "javascript", "react", "angular", "aws", "cloud", "sql", "nosql", "kubernetes", "docker"]
        for skill in skill_keywords:
            if skill in job_description.lower():
                if "required" in job_description.lower() or "must have" in job_description.lower():
                    required_skills.append(skill)
                else:
                    preferred_skills.append(skill)
        
        # Extract experience requirements
        exp_match = re.search(r'(\d+)\+?\s*years', job_description.lower())
        if exp_match:
            min_experience = int(exp_match.group(1))
            
        # Extract education requirements
        edu_levels = ["bachelor", "master", "phd", "doctorate", "high school"]
        for level in edu_levels:
            if level in job_description.lower():
                if level == "phd" or level == "doctorate":
                    required_education = "phd"
                elif level == "master":
                    required_education = "master's"
                elif level == "high school":
                    required_education = "high school"
                break
    
    # Calculate metrics for each candidate
    precision_scores = []
    skill_coverage_scores = []
    experience_match_scores = []
    education_match_scores = []
    
    for candidate in candidates_list:
        # For metrics that need the actual analysis results
        if "analysis" in candidate:
            analysis = candidate["analysis"]
            
            # Skill coverage
            candidate_skills = analysis.get("skills", [])
            required_match = sum(1 for skill in required_skills if skill in candidate_skills)
            preferred_match = sum(1 for skill in preferred_skills if skill in candidate_skills)
            
            required_coverage = required_match / max(1, len(required_skills))
            preferred_coverage = preferred_match / max(1, len(preferred_skills))
            
            skill_coverage = 0.7 * required_coverage + 0.3 * preferred_coverage
            skill_coverage_scores.append(skill_coverage)
            
            # Experience match
            years = analysis.get("years_experience", 0)
            experience_match = 1.0 if years >= min_experience else years / max(1, min_experience)
            experience_match_scores.append(experience_match)
            
            # Education match
            education = analysis.get("education_level", "").lower()
            education_levels = {"high school": 1, "associate's": 2, "bachelor's": 3, "master's": 4, "phd": 5}
            
            candidate_edu_level = education_levels.get(education, 3)  # Default to bachelor's if unknown
            required_edu_level = education_levels.get(required_education.lower(), 3)
            
            education_match = 1.0 if candidate_edu_level >= required_edu_level else 0.5  # Full match or partial match
            education_match_scores.append(education_match)
            
            # Calculate precision for this candidate
            weighted_score = 0.5 * skill_coverage + 0.3 * experience_match + 0.2 * education_match
            precision_scores.append(weighted_score)
        else:
            # For candidates without detailed analysis, estimate based on available info
            # This is a simplified approach - in a real system, you'd analyze the resume text
            candidate_skills = candidate.get("skills", candidate.get("Skills", []))
            
            # Estimate skill coverage
            required_match = sum(1 for skill in required_skills if any(req_skill.lower() == skill.lower() for req_skill in candidate_skills))
            preferred_match = sum(1 for skill in preferred_skills if any(pref_skill.lower() == skill.lower() for pref_skill in candidate_skills))
            
            required_coverage = required_match / max(1, len(required_skills))
            preferred_coverage = preferred_match / max(1, len(preferred_skills))
            
            skill_coverage = 0.7 * required_coverage + 0.3 * preferred_coverage
            skill_coverage_scores.append(skill_coverage)
            
            # Other scores get default values
            experience_match_scores.append(0.7)  # Assume moderate match
            education_match_scores.append(0.8)  # Assume good match
            
            # Calculate precision
            precision_scores.append(0.5 * skill_coverage + 0.3 * 0.7 + 0.2 * 0.8)
    
    # Calculate overall metrics
    if precision_scores:
        metrics["shortlisting_precision"] = sum(precision_scores) / len(precision_scores) * 100
    if skill_coverage_scores:
        metrics["skill_coverage"] = sum(skill_coverage_scores) / len(skill_coverage_scores) * 100
    if experience_match_scores:
        metrics["experience_match"] = sum(experience_match_scores) / len(experience_match_scores) * 100
    if education_match_scores:
        metrics["education_match"] = sum(education_match_scores) / len(education_match_scores) * 100
        
    # Calculate overall accuracy as weighted average
    metrics["overall_accuracy"] = (
        0.4 * metrics["shortlisting_precision"] +
        0.3 * metrics["skill_coverage"] +
        0.2 * metrics["experience_match"] +
        0.1 * metrics["education_match"]
    )
    
    # Update the class evaluation metrics
    self.evaluation_metrics["shortlisting_accuracy"] = metrics["overall_accuracy"]
    
    return metrics

  def evaluate_contact_extraction_efficiency(self, candidate_list):
    """
    Evaluate the efficiency of contact information extraction
    
    Args:
        candidate_list: List of candidates to evaluate
        
    Returns:
        Extraction efficiency metrics
    """
    metrics = {
        "overall_efficiency": 0,
        "email_extraction": 0,
        "phone_extraction": 0,
        "name_extraction": 0
    }
    
    if not candidate_list:
        return metrics
        
    # Count successful extractions
    email_extractions = 0
    phone_extractions = 0
    name_extractions = 0
    total_candidates = len(candidate_list)
    
    for candidate in candidate_list:
        # Check if candidate is a string (raw resume) or a dictionary
        if isinstance(candidate, str):
            # Extract email
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            email_match = re.search(email_pattern, candidate)
            if email_match:
                email_extractions += 1
                
            # Extract phone
            phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            phone_match = re.search(phone_pattern, candidate)
            if phone_match:
                phone_extractions += 1
                
            # Extract name (simplified - in real system would be more robust)
            name_pattern = r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s*\n)'
            name_match = re.search(name_pattern, candidate)
            if name_match:
                name_extractions += 1
        else:
            # If it's already a dictionary, check for extracted fields
            if "email" in candidate or "Email" in candidate:
                email_extractions += 1
            
            if "phone" in candidate or "Phone" in candidate:
                phone_extractions += 1
            
            if ("name" in candidate and candidate["name"] != "Unknown") or \
               ("Name" in candidate and candidate["Name"] != "Unknown"):
                name_extractions += 1
    
    # Calculate metrics
    if total_candidates > 0:
        metrics["email_extraction"] = (email_extractions / total_candidates) * 100
        metrics["phone_extraction"] = (phone_extractions / total_candidates) * 100
        metrics["name_extraction"] = (name_extractions / total_candidates) * 100
        
        # Overall efficiency is weighted average
        metrics["overall_efficiency"] = (
            0.4 * metrics["email_extraction"] +
            0.3 * metrics["phone_extraction"] +
            0.3 * metrics["name_extraction"]
        )
    
    # Update class evaluation metrics
    self.evaluation_metrics["contact_extraction_efficiency"] = metrics["overall_efficiency"]
    
    return metrics

  def evaluate_invitation_correctness(self, invitation_dict, interviews):
    """
    Evaluate the correctness and quality of generated interview invitations
    
    Args:
        invitation_dict: Dictionary mapping candidate IDs to invitation texts
        interviews: List of interview details
        
    Returns:
        Invitation correctness metrics
    """
    metrics = {
        "overall_correctness": 0,
        "info_completeness": 0,
        "personalization": 0,
        "clarity": 0
    }
    
    if not invitation_dict or not interviews:
        return metrics
        
    # Metrics for each invitation
    info_completeness_scores = []
    personalization_scores = []
    clarity_scores = []
    
    for interview in interviews:
        candidate_id = interview.get("candidate_id", "")
        if candidate_id in invitation_dict:
            invitation = invitation_dict[candidate_id]
            
            # Info completeness: Check for required information
            completeness_score = 0
            required_elements = [
                # Candidate name
                interview.get("candidate_name", ""),
                # Date
                interview.get("interview_date", "") or interview.get("date", ""),
                # Time
                interview.get("start_time", ""),
                # Interviewer
                interview.get("interviewer_name", ""),
                # Meeting link
                interview.get("meeting_link", "") or "meet.google.com"
            ]
            
            # Count how many required elements are included
            for element in required_elements:
                if element and element in invitation:
                    completeness_score += 1
            
            # Convert to percentage
            completeness_score = (completeness_score / len(required_elements)) * 100
            info_completeness_scores.append(completeness_score)
            
            # Personalization: Check for candidate-specific details
            personalization_score = 0
            
            # Check for greeting with name
            if f"Dear {interview.get('candidate_name', '')}" in invitation:
                personalization_score += 40
            elif "Dear" in invitation:  # Generic greeting
                personalization_score += 20
                
            # Check for skill mentions
            candidate_skills = interview.get("candidate_key_skills", [])
            skill_mentions = sum(1 for skill in candidate_skills if skill in invitation)
            personalization_score += min(40, skill_mentions * 10)  # Up to 40 points for skill mentions
            
            # Check for role-specific content
            role_keywords = ["position", "role", "opportunity", "job"]
            if any(keyword in invitation.lower() for keyword in role_keywords):
                personalization_score += 20
                
            personalization_scores.append(personalization_score)
            
            # Clarity: Check for clear instructions
            clarity_score = 0
            
            # Check for preparation instructions
            preparation_keywords = ["prepare", "review", "ready", "bring"]
            if any(keyword in invitation.lower() for keyword in preparation_keywords):
                clarity_score += 30
                
            # Check for next steps
            next_steps_keywords = ["next steps", "confirm", "response", "reply", "questions", "contact"]
            if any(keyword in invitation.lower() for keyword in next_steps_keywords):
                clarity_score += 30
                
            # Check for technical setup instructions
            setup_keywords = ["link", "join", "connect", "video", "audio", "camera", "microphone"]
            if any(keyword in invitation.lower() for keyword in setup_keywords):
                clarity_score += 40
                
            clarity_scores.append(clarity_score)
    
    # Calculate overall metrics
    if info_completeness_scores:
        metrics["info_completeness"] = sum(info_completeness_scores) / len(info_completeness_scores)
    if personalization_scores:
        metrics["personalization"] = sum(personalization_scores) / len(personalization_scores)
    if clarity_scores:
        metrics["clarity"] = sum(clarity_scores) / len(clarity_scores)
        
    # Overall correctness is weighted average
    metrics["overall_correctness"] = (
        0.4 * metrics["info_completeness"] +
        0.3 * metrics["personalization"] +
        0.3 * metrics["clarity"]
    )
    
    # Update class evaluation metrics
    self.evaluation_metrics["invitation_correctness"] = metrics["overall_correctness"]
    
    return metrics

  def evaluate_scheduling_fairness(self, interview_assignments):
    """
    Evaluate the fairness and optimization of interview scheduling
    
    Args:
        interview_assignments: List of interview assignments
        
    Returns:
        Scheduling fairness metrics
    """
    metrics = {
        "overall_fairness": 0,
        "workload_balance": 0,
        "expertise_utilization": 0,
        "time_efficiency": 0
    }
    
    if not interview_assignments:
        return metrics
    
    # Group interviews by interviewer
    interviewer_workloads = defaultdict(list)
    for interview in interview_assignments:
        interviewer_id = interview.get("interviewer_id", "")
        interviewer_workloads[interviewer_id].append(interview)
    
    # Calculate workload balance
    interviewer_counts = [len(interviews) for interviews in interviewer_workloads.values()]
    if interviewer_counts:
        avg_workload = sum(interviewer_counts) / len(interviewer_counts)
        max_workload = max(interviewer_counts)
        min_workload = min(interviewer_counts)
        
        # Perfect balance = 100, decreases as imbalance increases
        workload_balance = 100 - (10 * (max_workload - min_workload))
        metrics["workload_balance"] = max(0, workload_balance)
    
    # Calculate expertise utilization
    expertise_scores = []
    for interview in interview_assignments:
        expertise_match_score = interview.get("expertise_match_score", 0)
        expertise_scores.append(expertise_match_score)
    
    if expertise_scores:
        metrics["expertise_utilization"] = sum(expertise_scores) / len(expertise_scores)
    
    # Calculate time efficiency
    time_efficient_interviews = 0
    for interviewer_id, interviews in interviewer_workloads.items():
        # Sort interviews by time
        sorted_interviews = sorted(interviews, key=lambda x: x.get("start_time", "00:00"))
        
        # Check for adequate spacing between interviews
        for i in range(1, len(sorted_interviews)):
            prev_end = sorted_interviews[i-1].get("end_time", "")
            curr_start = sorted_interviews[i].get("start_time", "")
            
            if prev_end and curr_start:
                # Convert to minutes since midnight
                prev_end_parts = prev_end.split(":")
                curr_start_parts = curr_start.split(":")
                
                try:
                    prev_end_minutes = int(prev_end_parts[0]) * 60 + int(prev_end_parts[1])
                    curr_start_minutes = int(curr_start_parts[0]) * 60 + int(curr_start_parts[1])
                    
                    # Check if there's at least 15 minutes between interviews
                    if curr_start_minutes - prev_end_minutes >= 15:
                        time_efficient_interviews += 1
                except (ValueError, IndexError):
                    pass
        
        # Add first interview as efficient by default
        if interviews:
            time_efficient_interviews += 1
    
    # Calculate time efficiency as percentage of efficient transitions
    total_transitions = sum(max(0, len(interviews) - 1) for interviews in interviewer_workloads.values()) + len(interviewer_workloads)
    if total_transitions > 0:
        metrics["time_efficiency"] = (time_efficient_interviews / total_transitions) * 100
    
    # Calculate overall fairness as weighted average
    metrics["overall_fairness"] = (
        0.4 * metrics["workload_balance"] +
        0.4 * metrics["expertise_utilization"] +
        0.2 * metrics["time_efficiency"]
    )
    
    # Update class evaluation metrics
    self.evaluation_metrics["scheduling_fairness"] = metrics["overall_fairness"]
    
    return metrics

  def get_evaluation_metrics(self):
    """
    Get the current evaluation metrics for the system
    
    Returns:
        Dictionary with all evaluation metrics
    """
    return self.evaluation_metrics