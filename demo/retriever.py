import sys, time
sys.dont_write_bytecode = True

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import re
import json
import numpy as np
from collections import defaultdict

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.agent import AgentFinish
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.schema import SystemMessage, HumanMessage


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
    """
    List of IDs of the applicants to retrieve resumes for
    """
    id_list: List[str] = Field(..., description="List of IDs of the applicants to retrieve resumes for")


class JobDescription(BaseModel):
    """
    Descriptions of a job to retrieve similar resumes for
    """
    job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for") 


class JobCriteria(BaseModel):
    """
    Job criteria including category-specific candidate counts
    """
    job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for")
    categories: Dict[str, int] = Field(..., description="Dictionary mapping job categories to number of candidates to select")


class CandidateRankingCriteria(BaseModel):
    """
    Criteria for ranking candidates
    """
    skill_weight: float = Field(0.6, description="Weight for skill match score (0-1)")
    experience_weight: float = Field(0.3, description="Weight for experience match score (0-1)")
    education_weight: float = Field(0.1, description="Weight for education match score (0-1)")
    required_skills: List[str] = Field([], description="List of required skills for the position")
    preferred_skills: List[str] = Field([], description="List of preferred but not required skills")
    min_experience_years: int = Field(0, description="Minimum years of experience required")
    education_level: str = Field("Bachelor's", description="Minimum education level required")


class RAGRetriever():
    def __init__(self, vectorstore_db, df):
        self.vectorstore = vectorstore_db
        self.df = df
        # Skills taxonomy for matching related skills
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

    def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
        """
        Implement Reciprocal Rank Fusion for combining multiple ranked lists
        
        Args:
            document_rank_list: List of dictionaries mapping document IDs to scores
            k: Constant to prevent division by zero and control impact of high rankings
            
        Returns:
            Dictionary of document IDs to fused scores, sorted by score
        """
        fused_scores = {}
        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                fused_scores[doc] += 1 / (rank + k)
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        return reranked_results

    def __retrieve_docs_id__(self, question: str, k=50):
        """
        Retrieve documents from vector store with similarity scores
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            
        Returns:
            Dictionary mapping document IDs to similarity scores
        """
        try:
            docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
            docs_score = {str(doc.metadata.get("ID", "unknown")): score for doc, score in docs_score}
            return docs_score
        except Exception as e:
            print(f"Error in __retrieve_docs_id__: {str(e)}")
            return {}

    def retrieve_id_and_rerank(self, subquestion_list: list):
        """
        Retrieve documents for multiple subquestions and rerank using reciprocal rank fusion
        
        Args:
            subquestion_list: List of query strings
            
        Returns:
            Dictionary of document IDs to fused scores
        """
        document_rank_list = []
        for subquestion in subquestion_list:
            doc_scores = self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD)
            if doc_scores:  # Only add if we got results
                document_rank_list.append(doc_scores)
        
        # If we have document rankings, rerank them
        if document_rank_list:
            reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
            return reranked_documents
        else:
            # Fallback if no document rankings
            return {}

    def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
        """
        Retrieve full document content for a list of document IDs
        
        Args:
            doc_id_with_score: Dictionary mapping document IDs to scores
            threshold: Maximum number of documents to retrieve
            
        Returns:
            List of formatted document strings
        """
        # Default to an empty DataFrame if none provided
        if self.df is None or (hasattr(self.df, 'empty') and self.df.empty):
            print("Warning: Empty DataFrame in retrieve_documents_with_id")
            return []
            
        # Ensure we have a DataFrame for dictionary access
        if hasattr(self.df, 'to_dict'):
            # Ensure we have string IDs in our dataframe comparison
            try:
                id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
            except Exception as e:
                print(f"Error creating ID-resume dictionary: {str(e)}")
                # Try a different approach if the above fails
                id_resume_dict = {}
                try:
                    for _, row in self.df.iterrows():
                        id_resume_dict[str(row.get("ID", ""))] = row.get("Resume", "")
                except Exception as e2:
                    print(f"Error iterating DataFrame: {str(e2)}")
                    return []
        else:
            # Assume df is already a list of dictionaries
            id_resume_dict = {str(item.get("ID", "")): item.get("Resume", "") for item in self.df}
        
        retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
        retrieved_documents = []
        
        for id in retrieved_ids:
            if id in id_resume_dict:
                resume_text = id_resume_dict[id]
                formatted_doc = f"Applicant ID {id}\nName: Unknown\nSkills: Unknown\n{resume_text}"
                retrieved_documents.append(formatted_doc)
        
        # Ensure we return at least one document if available and no results were found
        if not retrieved_documents and id_resume_dict:
            # Fallback to first document in dictionary
            first_id = next(iter(id_resume_dict), None)
            if first_id:
                resume_text = id_resume_dict[first_id]
                formatted_doc = f"Applicant ID {first_id}\nName: Unknown\nSkills: Unknown\n{resume_text}"
                retrieved_documents.append(formatted_doc)
            
        return retrieved_documents

    def extract_skills_from_resume(self, resume_text: str) -> List[str]:
        """
        Extract skills from a resume text using keyword matching against our taxonomy
        
        Args:
            resume_text: The resume text to analyze
            
        Returns:
            List of extracted skills
        """
        extracted_skills = []
        
        # First look for skill section
        skill_section_patterns = [
            r"(?i)skills[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)",
            r"(?i)technical skills[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)",
            r"(?i)core competencies[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)"
        ]
        
        skill_section = None
        for pattern in skill_section_patterns:
            match = re.search(pattern, resume_text)
            if match:
                skill_section = match.group(1)
                break
        
        # Get all skills from our taxonomy
        all_skills = []
        for category_skills in self.skills_taxonomy.values():
            all_skills.extend(category_skills)
        
        # If we found a skills section, prioritize those skills
        if skill_section:
            for skill in all_skills:
                # Use word boundary to ensure we match whole words
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, skill_section, re.IGNORECASE):
                    extracted_skills.append(skill)
        
        # Also look for skills throughout the document
        for skill in all_skills:
            if skill not in extracted_skills:  # Avoid duplicates
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, resume_text, re.IGNORECASE):
                    extracted_skills.append(skill)
        
        return extracted_skills

    def extract_experience_years(self, resume_text: str) -> int:
        """
        Estimate years of experience from a resume
        
        Args:
            resume_text: The resume text to analyze
            
        Returns:
            Estimated years of experience (integer)
        """
        # Look for work experience section
        experience_section_patterns = [
            r"(?i)work experience[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)",
            r"(?i)experience[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)",
            r"(?i)professional experience[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)"
        ]
        
        experience_section = None
        for pattern in experience_section_patterns:
            match = re.search(pattern, resume_text)
            if match:
                experience_section = match.group(1)
                break
        
        # Look for date ranges in the resume or specifically in the experience section
        text_to_search = experience_section if experience_section else resume_text
        
        # Pattern to find date ranges like "2018 - 2023" or "Jan 2018 - Dec 2023" or "2018 - Present"
        date_patterns = [
            r'(\d{4})\s*[-–—to]\s*(\d{4}|\bpresent\b|\bcurrent\b)',
            r'(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+)?(\d{4})\s*[-–—to]\s*(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+)?(\d{4}|\bpresent\b|\bcurrent\b)'
        ]
        
        current_year = 2025  # Using the current year from the prompt
        total_years = 0
        found_dates = set()
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text_to_search, re.IGNORECASE):
                start_year = int(match.group(1))
                
                if match.group(2).lower() in ('present', 'current'):
                    end_year = current_year
                else:
                    try:
                        end_year = int(match.group(2))
                    except ValueError:
                        continue
                
                # Check for reasonable years (avoid parsing errors)
                if 1970 <= start_year <= current_year and start_year <= end_year:
                    # Avoid duplicate date ranges
                    date_key = f"{start_year}-{end_year}"
                    if date_key not in found_dates:
                        found_dates.add(date_key)
                        total_years += min(end_year - start_year, 7)  # Cap each job at 7 years
        
        # If no date ranges found, try to extract years of experience from statements
        if total_years == 0:
            year_patterns = [
                r'(\d+)\+?\s*(?:years|yrs)(?:\s+of)?\s+experience',
                r'experience\s+of\s+(\d+)\+?\s*(?:years|yrs)',
                r'(?:over|more\s+than)\s+(\d+)\s*(?:years|yrs)'
            ]
            
            for pattern in year_patterns:
                match = re.search(pattern, resume_text, re.IGNORECASE)
                if match:
                    try:
                        years = int(match.group(1))
                        if 0 < years < 50:  # Sanity check
                            total_years = years
                            break
                    except ValueError:
                        continue
        
        # Default to 2 years if we couldn't extract experience
        return max(1, total_years) if total_years > 0 else 2

    def extract_education_level(self, resume_text: str) -> str:
        """
        Extract highest education level from a resume
        
        Args:
            resume_text: The resume text to analyze
            
        Returns:
            Highest education level
        """
        education_levels = {
            "phd": ["phd", "ph.d", "doctor of philosophy", "doctorate"],
            "masters": ["master", "ms ", "m.s", "msc", "m.sc", "ma ", "m.a", "mba", "m.b.a"],
            "bachelors": ["bachelor", "bs ", "b.s", "ba ", "b.a", "b.tech", "b.e", "btech", "be "],
            "associates": ["associate", "a.s", "a.a"],
            "high school": ["high school", "secondary school", "hs diploma"]
        }
        
        # Look for education section
        education_section_patterns = [
            r"(?i)education[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)",
            r"(?i)academic background[\s:]+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n(?:[A-Z][a-zA-Z\s]+):|$)"
        ]
        
        education_section = None
        for pattern in education_section_patterns:
            match = re.search(pattern, resume_text)
            if match:
                education_section = match.group(1)
                break
        
        text_to_search = education_section if education_section else resume_text
        
        # Start with lowest education and work up to find highest
        highest_level = "high school"  # Default
        
        for level, keywords in education_levels.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword), text_to_search, re.IGNORECASE):
                    # Update if this is higher than what we've found
                    if level == "phd":
                        return "PhD"
                    elif level == "masters" and highest_level not in ["phd"]:
                        highest_level = "Masters"
                    elif level == "bachelors" and highest_level not in ["phd", "masters"]:
                        highest_level = "Bachelors"
                    elif level == "associates" and highest_level not in ["phd", "masters", "bachelors"]:
                        highest_level = "Associates"
        
        return highest_level

    def calculate_skill_match_score(self, candidate_skills, required_skills, preferred_skills):
        """
        Calculate a skill match score based on required and preferred skills
        
        Args:
            candidate_skills: List of candidate's skills
            required_skills: List of required skills for the position
            preferred_skills: List of preferred skills for the position
            
        Returns:
            Skill match score from 0-100
        """
        if not required_skills and not preferred_skills:
            return 50  # Default middle score if no skills specified
        
        # Convert all to lowercase for case-insensitive matching
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        required_skills_lower = [s.lower() for s in required_skills]
        preferred_skills_lower = [s.lower() for s in preferred_skills]
        
        # Calculate direct matches
        required_matches = sum(1 for skill in required_skills_lower if skill in candidate_skills_lower)
        preferred_matches = sum(1 for skill in preferred_skills_lower if skill in candidate_skills_lower)
        
        # Calculate related skill matches
        required_related = 0
        preferred_related = 0
        
        for req_skill in required_skills_lower:
            if req_skill not in candidate_skills_lower:
                # Check if candidate has related skills
                req_category = self.skill_to_category.get(req_skill)
                if req_category:
                    for cand_skill in candidate_skills_lower:
                        if self.skill_to_category.get(cand_skill) == req_category:
                            required_related += 0.5  # Half point for related skill
                            break
        
        for pref_skill in preferred_skills_lower:
            if pref_skill not in candidate_skills_lower:
                # Check if candidate has related skills
                pref_category = self.skill_to_category.get(pref_skill)
                if pref_category:
                    for cand_skill in candidate_skills_lower:
                        if self.skill_to_category.get(cand_skill) == pref_category:
                            preferred_related += 0.5  # Half point for related skill
                            break
        
        # Calculate scores
        required_weight = 0.7  # 70% weight for required skills
        preferred_weight = 0.3  # 30% weight for preferred skills
        
        # Handle division by zero
        if len(required_skills_lower) > 0:
            required_score = 100 * (required_matches + required_related) / len(required_skills_lower)
        else:
            required_score = 100 if len(candidate_skills_lower) > 0 else 0
            
        if len(preferred_skills_lower) > 0:
            preferred_score = 100 * (preferred_matches + preferred_related) / len(preferred_skills_lower)
        else:
            preferred_score = 50  # Neutral score for preferred skills if none specified
        
        # Calculate total score
        total_score = required_weight * required_score + preferred_weight * preferred_score
        
        return total_score

    def calculate_experience_match_score(self, years_experience, min_years_required):
        """
        Calculate experience match score based on years of experience
        
        Args:
            years_experience: Candidate's years of experience
            min_years_required: Minimum years required for the position
            
        Returns:
            Experience match score from 0-100
        """
        if years_experience < min_years_required:
            # Partial credit if close to minimum
            if years_experience >= 0.7 * min_years_required:
                return 60 * (years_experience / min_years_required)
            else:
                return 40 * (years_experience / min_years_required)
        elif years_experience == min_years_required:
            return 80  # Exact match
        elif years_experience <= 1.5 * min_years_required:
            # More than minimum but not too much
            return 90 + 10 * ((years_experience - min_years_required) / (0.5 * min_years_required))
        else:
            # Significantly more experience than required
            # Could be overqualified, but still a good match
            return 95
            
    def calculate_education_match_score(self, education_level, required_level):
        """
        Calculate education match score based on education level
        
        Args:
            education_level: Candidate's education level
            required_level: Required education level for the position
            
        Returns:
            Education match score from 0-100
        """
        # Education level hierarchy
        levels = {
            "high school": 1,
            "associates": 2,
            "bachelors": 3,
            "masters": 4,
            "phd": 5
        }
        
        # Normalize education level strings
        education_level = education_level.lower()
        required_level = required_level.lower()
        
        # Handle common abbreviations
        for full_name, level_value in levels.items():
            if education_level.startswith(full_name[:3]):
                education_level = full_name
            if required_level.startswith(full_name[:3]):
                required_level = full_name
        
        # Get numeric values for levels
        candidate_level = levels.get(education_level, 1)  # Default to high school if not found
        required_level_value = levels.get(required_level, 3)  # Default to bachelor's if not found
        
        if candidate_level < required_level_value:
            # Below required level
            deficit = required_level_value - candidate_level
            if deficit == 1:
                return 60  # One level below
            else:
                return max(30, 70 - (deficit * 20))  # Decrease score for each level below
        elif candidate_level == required_level_value:
            return 90  # Exact match
        else:
            # Above required level
            return 95  # Slightly higher for more education than required
            
    def analyze_resume(self, resume_text, job_description):
        """
        Analyze a resume against a job description
        
        Args:
            resume_text: The resume text to analyze
            job_description: The job description to compare against
            
        Returns:
            Dictionary with analysis results
        """
        # Extract candidate information
        extracted_skills = self.extract_skills_from_resume(resume_text)
        years_experience = self.extract_experience_years(resume_text)
        education_level = self.extract_education_level(resume_text)
        
        # Extract job requirements - simplified version
        # In a production system, this would use more sophisticated NLP
        job_skills = self.extract_skills_from_resume(job_description)  # Reuse skill extractor
        
        # Determine required vs preferred skills
        # For this simple implementation, we'll consider first 60% of skills as required
        required_count = max(1, int(len(job_skills) * 0.6))
        required_skills = job_skills[:required_count]
        preferred_skills = job_skills[required_count:]
        
        # Estimate minimum experience from job description
        experience_patterns = [
            r'(\d+)\+?\s*(?:years|yrs)(?:\s+of)?\s+experience',
            r'experience\s+of\s+(\d+)\+?\s*(?:years|yrs)',
            r'(?:over|more\s+than)\s+(\d+)\s*(?:years|yrs)'
        ]
        
        min_years_required = 1  # Default
        for pattern in experience_patterns:
            match = re.search(pattern, job_description, re.IGNORECASE)
            if match:
                try:
                    years = int(match.group(1))
                    if 0 < years < 20:  # Sanity check
                        min_years_required = years
                        break
                except ValueError:
                    continue
        
        # Determine required education level
        education_keywords = {
            "phd": ["phd", "ph.d", "doctor of philosophy", "doctorate"],
            "masters": ["master", "ms ", "m.s", "msc", "m.sc", "ma ", "m.a", "mba", "m.b.a"],
            "bachelors": ["bachelor", "bs ", "b.s", "ba ", "b.a", "b.tech", "b.e"],
            "associates": ["associate", "a.s", "a.a"]
        }
        
        required_education = "bachelors"  # Default
        for level, keywords in education_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword), job_description, re.IGNORECASE):
                    if level == "phd":
                        required_education = "phd"
                        break
                    elif level == "masters" and required_education != "phd":
                        required_education = "masters"
                    elif level == "bachelors" and required_education not in ["phd", "masters"]:
                        required_education = "bachelors"
                    elif level == "associates" and required_education not in ["phd", "masters", "bachelors"]:
                        required_education = "associates"
            if required_education == "phd":
                break
                
        # Calculate scores
        skill_score = self.calculate_skill_match_score(extracted_skills, required_skills, preferred_skills)
        experience_score = self.calculate_experience_match_score(years_experience, min_years_required)
        education_score = self.calculate_education_match_score(education_level, required_education)
        
        # Store analysis results
        return {
            "skills": extracted_skills,
            "years_experience": years_experience,
            "education_level": education_level,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "skill_score": skill_score,
            "experience_score": experience_score,
            "education_score": education_score,
            "min_years_required": min_years_required,
            "required_education": required_education
        }

    def calculate_overall_score(self, analysis, weights=None):
        """
        Calculate overall candidate score based on weighted components
        
        Args:
            analysis: Analysis results from analyze_resume
            weights: Dictionary with weights for each component
            
        Returns:
            Overall score from 0-100
        """
        # Default weights
        if weights is None:
            weights = {
                "skill": 0.5,
                "experience": 0.3,
                "education": 0.2
            }
        
        # Calculate weighted score
        skill_contribution = analysis["skill_score"] * weights["skill"]
        experience_contribution = analysis["experience_score"] * weights["experience"]
        education_contribution = analysis["education_score"] * weights["education"]
        
        overall_score = skill_contribution + experience_contribution + education_contribution
        
        return overall_score

    def generate_llm_evaluation(self, resume_text, job_query, llm):
      """
      Generate LLM-based evaluation of a resume's match to job requirements
      
      Args:
          resume_text: The candidate's resume text
          job_query: The original job description or query
          llm: Language model instance for evaluation
          
      Returns:
          Dictionary with evaluation results for contextual relevancy, faithfulness, and fairness
      """
      if not llm:
          return {
              "contextual_relevancy": "Not evaluated (LLM unavailable)",
              "faithfulness": "Not evaluated (LLM unavailable)",
              "fairness": "Not evaluated (LLM unavailable)"
          }
      
      # Prepare a clean, truncated version of the resume to avoid token limits
      resume_sample = resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text
      
      # Create the evaluation prompt
      prompt = f"""
      You are an expert HR evaluator assessing candidate-job fit. Evaluate this candidate's resume against the job requirements.
      
      Job Requirements: {job_query}
      
      Candidate Resume: {resume_sample}
      
      Please evaluate on these three metrics:
      
      1. Contextual Relevancy (1-100): How relevant is the candidate's experience and skills to these specific job requirements?
      
      2. Faithfulness (1-100): How accurately does the candidate's resume address the core requirements? Are there any mismatches or gaps?
      
      3. Fairness (1-100): Is your evaluation balanced and unbiased? Consider all aspects of the candidate's qualifications.
      
      For each metric, provide a score and a brief 1-3 sentence explanation along with the meaning of each score (what each score tries to evaluate).
      Format your response as:
      "Contextual Relevancy: [Score] - [Explanation]"
      "Faithfulness: [Score] - [Explanation]"
      "Fairness: [Score] - [Explanation]"
      """
      
      try:
          # Call the LLM with the prompt
          system_message = SystemMessage(content="You are an expert HR evaluator providing objective candidate assessments.")
          user_message = HumanMessage(content=prompt)
          response = llm.llm.invoke([system_message, user_message])
          
          # Parse the response
          eval_results = {
              "contextual_relevancy": "",
              "faithfulness": "",
              "fairness": ""
          }
          
          # Extract sections from response
          response_text = response.content
          for line in response_text.split('\n'):
              if "Contextual Relevancy:" in line:
                  eval_results["contextual_relevancy"] = line.replace("Contextual Relevancy:", "").strip()
              elif "Faithfulness:" in line:
                  eval_results["faithfulness"] = line.replace("Faithfulness:", "").strip()
              elif "Fairness:" in line:
                  eval_results["fairness"] = line.replace("Fairness:", "").strip()
          
          return eval_results
      except Exception as e:
          print(f"Error generating LLM evaluation: {e}")
          return {
              "contextual_relevancy": "Evaluation failed",
              "faithfulness": "Evaluation failed",
              "fairness": "Evaluation failed"
          }
        
    def generate_shortlisting_reason(self, analysis, overall_score, resume_text="", job_query="", llm=None):
      """
      Generate a human-readable explanation for why a candidate was shortlisted
      
      Args:
          analysis: Analysis results from analyze_resume
          overall_score: Overall candidate score
          resume_text: The candidate's resume text (optional)
          job_query: The original job query (optional)
          llm: Language model instance for evaluation (optional)
          
      Returns:
          Explanation string
      """
      reasons = []
      
      # Add skill match reason
      skill_match_percent = int(analysis["skill_score"])
      if skill_match_percent >= 80:
          matching_skills = [skill for skill in analysis["skills"] if skill in analysis["required_skills"]]
          if matching_skills:
              reasons.append(f"Strong skill match ({skill_match_percent}%) with key skills: {', '.join(matching_skills[:3])}")
          else:
              reasons.append(f"Strong overall skill profile ({skill_match_percent}%)")
      elif skill_match_percent >= 60:
          reasons.append(f"Good skill match ({skill_match_percent}%)")
      # else:
      #     reasons.append(f"Partial skill match ({skill_match_percent}%)")
      
      # Add experience reason
      years = analysis["years_experience"]
      required = analysis["min_years_required"]
      if years >= required:
          reasons.append(f"{years} years of experience (meets {required} year requirement)")
      else:
          reasons.append(f"{years} years of experience (below {required} year requirement)")
      
      # Add education reason
      edu_level = analysis["education_level"]
      req_edu = analysis["required_education"]
      edu_score = int(analysis["education_score"])
      
      if edu_score >= 90:
          reasons.append(f"Education ({edu_level}) meets or exceeds requirements ({req_edu})")
      elif edu_score >= 60:
          reasons.append(f"Education ({edu_level}) is close to requirements ({req_edu})")
      else:
          reasons.append(f"Education ({edu_level}) is below requirements ({req_edu})")
      
      # Create formatted explanation
      explanation = f"Overall score: {int(overall_score)}/100\n"
      explanation += "Reasons:\n- " + "\n- ".join(reasons)
      
      # Add LLM evaluation if provided
      if llm and resume_text and job_query:
          try:
              llm_eval = self.generate_llm_evaluation(resume_text, job_query, llm)
              
              explanation += "\n\n### LLM as a Judge Evals:"
              if llm_eval["contextual_relevancy"]:
                  explanation += f"\n- **Contextual Relevancy**: {llm_eval['contextual_relevancy']}"
              if llm_eval["faithfulness"]:
                  explanation += f"\n- **Faithfulness**: {llm_eval['faithfulness']}"
              if llm_eval["fairness"]:
                  explanation += f"\n- **Fairness**: {llm_eval['fairness']}"
          except Exception as e:
              print(f"Error adding LLM evaluation to shortlisting reason: {e}")
      
      return explanation


class SelfQueryRetriever(RAGRetriever):
    def __init__(self, vectorstore_db, df):
        super().__init__(vectorstore_db, df)

        # Document expected behavior examples
        self.query_examples = {
            "single_candidate": "Please find me a genai candidate",
            "specific_count": "Find 3 fullstack developers with React experience",
            "multi_category": "I need a fullstack developer and 2 genai candidates",
            "category_with_skills": "Find me 2 frontend developers who know React and 1 backend developer with Python experience",
            "interview_scheduling": "Schedule interviews with all the selected candidates on Thursday"
        }
        
        # Use a more explicit system prompt to encourage retrieval
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in talent acquisition and resume matching.
                        Your primary role is to help retrieve relevant candidate resumes.
                        
                        For job description queries: Always use the job description to retrieve matching resumes
                        For applicant ID queries: Extract the IDs to fetch specific resumes
                        
                        DO NOT respond with general advice - your job is to retrieve documents."""),
            ("user", "{input}")
        ])
        
        self.meta_data = {
            "rag_mode": "",
            "query_type": "retrieve_applicant_jd",  # Default to JD retrieval
            "extracted_input": "",
            "subquestion_list": [],
            "retrieved_docs_with_scores": [],
            "required_count": 1,  # Default to 1 candidate
            "analysis_results": [],  # Store detailed analysis of candidates
            "shortlisting_reasons": {},  # Store reasons for shortlisting
            "evaluation_metrics": {}  # Store evaluation metrics
        }

    def retrieve_docs(self, question: str, llm, rag_mode: str, k=None):
        """
        Retrieve relevant documents based on the question using the appropriate retrieval strategy
        
        Args:
            question: User query string
            llm: Language model instance for generating subqueries
            rag_mode: Retrieval mode (RAG or RAG Fusion)
            k: Number of documents to retrieve (defaults to None if not specified)
            
        Returns:
            List of retrieved document strings
        """
        # Extract role info
        role_info = self._extract_role_and_skills(question)
        self.meta_data["role_type"] = role_info.get("role_type", "")
        self.meta_data["skills"] = role_info.get("skills", [])
        self.meta_data["rag_mode"] = rag_mode
        
        # Extract categories first
        categories = self._extract_categories(question)
        if categories:
            self.meta_data["categories"] = categories
            
        # If we have categories, use that for retrieval
        if categories:
            # Use category-based retrieval
            self.meta_data["query_type"] = "retrieve_applicant_category"
            self.meta_data["extracted_input"] = question
            
            # Sum total required candidates across categories
            total_required = sum(categories.values())
            self.meta_data["required_count"] = total_required
            
            # Log the request details
            print(f"Category-based query - categories: {categories}, total: {total_required}")
            
            # Retrieve by categories
            return self._retrieve_by_categories(question, categories, rag_mode, llm)
        
        # For non-category queries, extract required count
        required_count = self._extract_required_count(question)
        if required_count is None:
            # If no specific count found, default to 1 for focused results
            required_count = 1
            
        self.meta_data["required_count"] = required_count
        
        # ---- Extract IDs if present ----
        id_pattern = r'(?:id|ID)[\s:]*#?\s*(\d+)'
        id_matches = re.findall(id_pattern, question)
        
        if id_matches:
            # This is an ID-based query
            self.meta_data["query_type"] = "retrieve_applicant_id"
            self.meta_data["extracted_input"] = id_matches
            
            result = self._retrieve_by_ids(id_matches)
            return result
        
        # ---- Handle job description based retrieval ----
        # Regular job description query
        self.meta_data["query_type"] = "retrieve_applicant_jd"
        self.meta_data["extracted_input"] = question
        
        # Generate subquestions if using RAG Fusion
        subquestion_list = [question]
        if rag_mode == "RAG Fusion":
            try:
                generated_questions = llm.generate_subquestions(question)
                if generated_questions:
                    subquestion_list.extend(generated_questions)
            except Exception as e:
                print(f"Error generating subquestions: {e}")
        
        self.meta_data["subquestion_list"] = subquestion_list
        
        # Retrieve documents based on subquestions
        retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
        self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
        
        # Get more documents than needed for better filtering
        buffer_count = max(required_count * 2, 10)
        retrieved_documents = self.retrieve_documents_with_id(retrieved_ids, threshold=buffer_count)
        
        # If we got limited results, try focused retrieval
        if len(retrieved_documents) < required_count:
            # Try role-based retrieval
            if role_info.get("role_type"):
                print(f"Limited results, trying role-based retrieval for '{role_info['role_type']}'")
                role_docs = self._retrieve_by_role_type(role_info["role_type"], buffer_count)
                
                # Add non-duplicates
                existing_ids = self._extract_ids_from_docs(retrieved_documents)
                for doc in role_docs:
                    doc_id = self._extract_id_from_doc(doc)
                    if doc_id and doc_id not in existing_ids:
                        retrieved_documents.append(doc)
                        existing_ids.add(doc_id)
            
            # Try skill-based retrieval if still needed
            if len(retrieved_documents) < required_count and role_info.get("skills"):
                print(f"Still need more results, trying skill-based retrieval")
                skill_docs = self._retrieve_by_skills(role_info["skills"], buffer_count)
                
                # Add non-duplicates
                existing_ids = self._extract_ids_from_docs(retrieved_documents)
                for doc in skill_docs:
                    doc_id = self._extract_id_from_doc(doc)
                    if doc_id and doc_id not in existing_ids:
                        retrieved_documents.append(doc)
                        existing_ids.add(doc_id)
            
        # If still no results, try direct vector search
        if not retrieved_documents:
            print("No results from retrieval, trying direct search")
            direct_docs = self.__retrieve_docs_id__(question, k=buffer_count)
            retrieved_documents = self.retrieve_documents_with_id(direct_docs, threshold=buffer_count)
        
        # Analyze and rank documents
        analyzed_docs = []
        shortlisting_reasons = {}
        
        for doc in retrieved_documents:
          doc_id = self._extract_id_from_doc(doc)
          if doc_id:
              try:
                  # Analyze the resume
                  analysis = self.analyze_resume(doc, question)
                  overall_score = self.calculate_overall_score(analysis)
                  
                  # Store the analysis
                  if "analysis_results" not in self.meta_data:
                      self.meta_data["analysis_results"] = {}
                  
                  self.meta_data["analysis_results"][doc_id] = {
                      "analysis": analysis,
                      "overall_score": overall_score
                  }
                  
                  # Generate reason with LLM evaluation
                  reason = self.generate_shortlisting_reason(
                      analysis, 
                      overall_score,
                      doc,  # Pass the full resume text
                      question,  # Pass the original query
                      llm  # Pass the LLM instance
                  )
                  shortlisting_reasons[doc_id] = {"reason": reason}
                  
                  # Add to ranking list
                  analyzed_docs.append((doc, overall_score, doc_id))
              except Exception as e:
                  print(f"Error analyzing document {doc_id}: {e}")
                  # Still include the document without analysis
                  analyzed_docs.append((doc, 0, doc_id))
        # Store shortlisting reasons
        self.meta_data["shortlisting_reasons"] = shortlisting_reasons
        
        # Sort by score and take top required_count
        analyzed_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _, _ in analyzed_docs[:required_count]]
        
        # If we don't have enough, add from remaining documents
        if len(top_docs) < required_count and len(retrieved_documents) > len(top_docs):
            # Get IDs of docs we already have
            top_doc_ids = set(self._extract_id_from_doc(doc) for doc in top_docs)
            
            # Add more docs until we reach required_count
            for doc in retrieved_documents:
                doc_id = self._extract_id_from_doc(doc)
                if doc_id and doc_id not in top_doc_ids:
                    top_docs.append(doc)
                    top_doc_ids.add(doc_id)
                
                if len(top_docs) >= required_count:
                    break
        
        # Add evaluation metrics
        selected_ids = [self._extract_id_from_doc(doc) for doc in top_docs]
        selected_candidates = {}
        
        for doc_id in selected_ids:
            if doc_id and doc_id in self.meta_data.get("analysis_results", {}):
                selected_candidates[doc_id] = self.meta_data["analysis_results"][doc_id]
        
        self.evaluate_resume_shortlisting(selected_candidates, required_count)
        
        return top_docs
    
    def _retrieve_by_ids(self, id_list):
        """Retrieve resumes by ID"""
        retrieved_resumes = []

        for id in id_list:
            try:
                if hasattr(self.df, 'loc'):
                    # DataFrame lookup
                    matching_rows = self.df[self.df["ID"].astype(str) == id]
                    if not matching_rows.empty:
                        resume_df = matching_rows.iloc[0]
                        resume_with_id = f"Applicant ID {resume_df['ID']}\nName: Unknown\nSkills: Unknown\n{resume_df['Resume']}"
                        retrieved_resumes.append(resume_with_id)
                else:
                    # Dictionary/list lookup
                    for item in self.df:
                        if str(item.get("ID", "")) == id:
                            resume_with_id = f"Applicant ID {item['ID']}\nName: Unknown\nSkills: Unknown\n{item['Resume']}"
                            retrieved_resumes.append(resume_with_id)
                            break
            except Exception as e:
                print(f"Error retrieving resume for ID {id}: {e}")
        
        # If no resumes found, provide a helpful error message
        if not retrieved_resumes:
            print(f"No resumes found for IDs: {id_list}")
            
        return retrieved_resumes
    
    def _retrieve_by_categories(self, job_description, categories, rag_mode, llm):
        """
        Retrieve candidates for multiple categories with enhanced accuracy
        
        Args:
            job_description: The job description or query text
            categories: Dictionary mapping category names to required counts
            rag_mode: Retrieval mode (RAG or RAG Fusion)
            llm: Language model for generating subqueries
            
        Returns:
            List of formatted candidates grouped by category
        """
        # First get subquestions for retrieval
        subquestions = {}
        base_subquestion_list = [job_description]
        
        # Create category-specific subquestions
        for category, count in categories.items():
            # Create a category-specific query
            category_query = f"{job_description} {category} experience skills expertise"
            
            # Add specific skills based on category
            if category == "genai":
                category_query += " generative AI language models LLM transformers prompt engineering"
            elif category == "fullstack":
                category_query += " full stack frontend backend web development JavaScript React Node.js"
            elif category == "frontend":
                category_query += " frontend UI UX design React Angular Vue JavaScript CSS"
            elif category == "backend":
                category_query += " backend server-side API database SQL Node.js Django"
                
            subquestions[category] = [category_query]
            base_subquestion_list.append(category_query)
        
        # If using RAG Fusion, generate additional subquestions
        if rag_mode == "RAG Fusion":
            try:
                # Generate general subquestions
                generated_questions = llm.generate_subquestions(job_description)
                base_subquestion_list.extend(generated_questions or [])
                
                # Generate category-specific subquestions
                for category in categories:
                    cat_query = f"{category} developer for {job_description}"
                    cat_subqs = llm.generate_subquestions(cat_query)
                    if cat_subqs:
                        subquestions[category].extend(cat_subqs)
            except Exception as e:
                print(f"Error generating subquestions for categories: {e}")
        
        # Store subquestions in metadata
        self.meta_data["subquestion_list"] = base_subquestion_list
        self.meta_data["category_subquestions"] = subquestions
        
        # Retrieve candidates for each category
        category_candidates = {}
        all_retrieved_ids = set()
        
        for category, count in categories.items():
            # Get category-specific subquestions
            category_queries = subquestions.get(category, [f"{category} {job_description}"])
            
            # Retrieve documents for this category
            retrieved_ids = self.retrieve_id_and_rerank(category_queries)
            
            # Store IDs for this category
            if category not in category_candidates:
                category_candidates[category] = []
                
            # Get the top scoring IDs for this category
            top_ids = list(sorted(retrieved_ids, key=retrieved_ids.get, reverse=True))[:count*3]  # Get more than needed for better filtering
            all_retrieved_ids.update(top_ids)
        
        # Retrieve actual resume texts
        candidates_by_id = {}
        
        if hasattr(self.df, 'loc'):
            # DataFrame approach
            try:
                for id in all_retrieved_ids:
                    matching_rows = self.df[self.df["ID"].astype(str) == id]
                    if not matching_rows.empty:
                        resume_text = matching_rows.iloc[0]["Resume"]
                        candidates_by_id[id] = resume_text
            except Exception as e:
                print(f"Error retrieving candidates from DataFrame: {e}")
        else:
            # Dictionary/list approach
            try:
                id_resume_dict = {str(item.get("ID", "")): item.get("Resume", "") for item in self.df}
                candidates_by_id = {id: id_resume_dict[id] for id in all_retrieved_ids if id in id_resume_dict}
            except Exception as e:
                print(f"Error retrieving candidates from dictionary: {e}")
        
        # Analyze each candidate against category-specific queries
        candidate_analyses = {}
        category_scores = {}
        
        for candidate_id, resume_text in candidates_by_id.items():
            try:
                # Analyze resume against general requirements
                analysis = self.analyze_resume(resume_text, job_description)
                
                # Calculate category-specific scores
                cat_scores = {}
                for category in categories:
                    # Create category-specific query for scoring
                    category_query = f"{category} developer experience with {' '.join(analysis.get('skills', []))}"
                    cat_relevance = self._calculate_role_relevance(resume_text, category, analysis.get('skills', []))
                    cat_scores[category] = cat_relevance
                
                # Find best matching category
                best_category = max(cat_scores, key=cat_scores.get) if cat_scores else list(categories.keys())[0]
                best_score = cat_scores.get(best_category, 0)
                
                # Store analysis
                candidate_analyses[candidate_id] = {
                    "analysis": analysis,
                    "overall_score": best_score,
                    "category_scores": cat_scores,
                    "resume": resume_text
                }
                
                # Organize by category, storing candidate_id and score
                for category, score in cat_scores.items():
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append((candidate_id, score))
                    
            except Exception as e:
                print(f"Error analyzing candidate {candidate_id}: {e}")
        
        # Store candidate analyses in metadata
        self.meta_data["analysis_results"] = candidate_analyses
        
        # Select top candidates for each category
        selected_candidates = {}
        already_assigned = set()
        
        # First pass: assign to best matching category
        for category, count in categories.items():
            if category in category_scores:
                # Sort by score (highest first)
                sorted_candidates = sorted(category_scores[category], key=lambda x: x[1], reverse=True)
                
                # Select top candidates that haven't been assigned yet
                selected_for_category = []
                for candidate_id, score in sorted_candidates:
                    if candidate_id not in already_assigned and len(selected_for_category) < count:
                        selected_for_category.append({
                            "id": candidate_id, 
                            "score": score,
                            "resume": candidates_by_id[candidate_id],
                            "analysis": candidate_analyses[candidate_id]["analysis"],
                            "category": category
                        })
                        already_assigned.add(candidate_id)
                
                selected_candidates[category] = selected_for_category
        
        # Store how many candidates we found for each category
        found_counts = {category: len(candidates) for category, candidates in selected_candidates.items()}
        self.meta_data["found_counts"] = found_counts
        
        # Generate shortlisting reasons
        shortlisting_reasons = {}
        for category, candidates_list in selected_candidates.items():
          for candidate in candidates_list:
              candidate_id = candidate["id"]
              if "analysis" in candidate:
                  reason = self.generate_shortlisting_reason(
                      candidate["analysis"], 
                      candidate["score"],
                      candidate["resume"],  # Pass the full resume text
                      job_description,  # Pass the original query
                      llm  # Pass the LLM instance
                  )
                  shortlisting_reasons[candidate_id] = {
                      "category": category,
                      "reason": reason
                  }
              
        # Store shortlisting reasons in metadata
        self.meta_data["shortlisting_reasons"] = shortlisting_reasons
        
        # Format the results for return
        retrieved_documents = []
        
        # Format each candidate with category information
        for category, candidates_list in selected_candidates.items():
            for candidate in candidates_list:
                try:
                    skills_text = ""
                    if "analysis" in candidate and "skills" in candidate["analysis"]:
                        skills_text = ", ".join(candidate["analysis"]["skills"])
                    
                    # Format with explicit category marking
                    resume_text = f"Applicant ID {candidate['id']}\nCategory: {category}\nName: Unknown\nSkills: {skills_text}\n{candidate['resume']}"
                    retrieved_documents.append(resume_text)
                except Exception as e:
                    print(f"Error formatting candidate {candidate.get('id', 'unknown')}: {e}")
                
        return retrieved_documents

    def _extract_required_count(self, query):
        """
        Extract the number of candidates requested with improved accuracy
        
        Args:
            query: The user query string
            
        Returns:
            Number of candidates requested, or None if not specified
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

    def _extract_categories(self, query):
        """Extract multiple job categories and counts from the query with enhanced pattern matching"""
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

    def _calculate_role_relevance(self, resume_text, role_type, skills):
        """
        Calculate how relevant a resume is to a specific role type
        
        Args:
            resume_text: Resume content
            role_type: Requested role type (e.g., "data scientist", "full stack")
            skills: List of requested skills
            
        Returns:
            Relevance score from 0-100
        """
        # Define role-specific keywords
        role_keywords = {
            "data scientist": ["data science", "machine learning", "ml", "ai", "statistics", 
                              "python", "r", "tensorflow", "pytorch", "scikit-learn", 
                              "data analysis", "data mining", "predictive modeling"],
            "data engineer": ["data engineering", "etl", "data pipeline", "data warehouse", 
                            "spark", "hadoop", "big data", "sql", "nosql", "data architecture"],
            "fullstack": ["full stack", "fullstack", "frontend", "backend", "web development", 
                          "javascript", "node.js", "react", "angular", "vue", "django", "flask"],
            "frontend": ["frontend", "ui", "ux", "react", "angular", "vue", "html", "css", "javascript"],
            "backend": ["backend", "server-side", "api", "database", "sql", "node.js", "django", "flask", "spring"],
            "devops": ["devops", "ci/cd", "continuous integration", "deployment", "docker", 
                      "kubernetes", "aws", "azure", "gcp", "cloud", "infrastructure"],
            "machine learning": ["machine learning", "ml", "neural networks", "deep learning", 
                                "nlp", "computer vision", "tensorflow", "pytorch"],
            "genai": ["generative ai", "genai", "ai", "artificial intelligence", "machine learning", "deep learning", 
                          "neural networks", "nlp", "transformers", "llm", "language model"]
        }
        
        # Normalize role type for matching
        normalized_role = role_type.lower() if role_type else ""
        
        # Find the most relevant role category based on the provided role_type
        best_role_match = None
        best_match_score = 0
        
        for role, keywords in role_keywords.items():
            # Check if our role_type is contained in this role category
            if normalized_role and (normalized_role in role or role in normalized_role):
                best_role_match = role
                best_match_score = 1.0
                break
                
            # Otherwise check for partial matches
            if normalized_role:
                for keyword in keywords:
                    if keyword in normalized_role:
                        best_role_match = role
                        best_match_score = 0.8
                        break
                    
        if not best_role_match and normalized_role:
            # If no direct match, find closest role by keyword overlap
            for role, keywords in role_keywords.items():
                # Count how many role keywords appear in our role_type
                match_count = sum(1 for keyword in keywords if keyword in normalized_role)
                match_score = match_count / len(keywords) if keywords else 0
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_role_match = role
        
        # If we still don't have a match, use the original role_type or default to general
        if not best_role_match:
            best_role_match = normalized_role if normalized_role else "general"
            
        # Count how many role-specific keywords are in the resume
        role_keyword_count = 0
        resume_lower = resume_text.lower()
        
        if best_role_match in role_keywords:
            for keyword in role_keywords[best_role_match]:
                if keyword in resume_lower:
                    role_keyword_count += 1
                    
        # Calculate role keyword relevance (0-60 points)
        role_relevance = 0
        if best_role_match in role_keywords:
            max_count = len(role_keywords[best_role_match])
            role_relevance = min(60, (role_keyword_count / max(1, max_count)) * 60)
            
        # Calculate required skills relevance (0-40 points)
        skill_relevance = 0
        if skills:
            skill_matches = sum(1 for skill in skills if skill.lower() in resume_lower)
            skill_relevance = (skill_matches / max(1, len(skills))) * 40
        else:
            skill_relevance = 20  # Neutral if no required skills specified
            
        # Return combined relevance score
        return role_relevance + skill_relevance

    def _extract_role_and_skills(self, query):
        """Extract role type and skills from the query"""
        role_type = ""
        skills = []
        
        # Extract role type
        role_patterns = [
            r"(?:for|with)\s+(?:a|an)?\s*([\w\s]+?)\s*(?:role|position|developer|engineer|designer|specialist)s?",
            r"([\w\s]+?)\s+developer",
            r"([\w\s]+?)\s+engineer",
            r"([\w\s]+?)\s+specialist"
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, query.lower())
            if match:
                role_type = match.group(1).strip()
                break
        
        # Extract skills
        common_skills = [
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php",
            "react", "angular", "vue", "node", "django", "flask", "spring",
            "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "terraform",
            "sql", "mysql", "postgresql", "mongodb", "nosql", "redis",
            "machine learning", "ai", "data science", "nlp", "genai",
            "frontend", "backend", "fullstack", "full stack", "mobile", "ios", "android",
            "devops", "cicd", "jenkins", "agile", "scrum"
        ]
        
        for skill in common_skills:
            if skill in query.lower():
                skills.append(skill)
        
        return {
            "role_type": role_type,
            "skills": skills
        }

    def _retrieve_by_role_type(self, role_type, k=5):
        """Retrieve resumes matching a specific role type"""
        # Create a query that emphasizes the role type
        role_query = f"{role_type} developer experience skills professional"
        
        # First try direct retrieval
        doc_ids = self.__retrieve_docs_id__(role_query, k=k*2)
        if doc_ids:
            return self.retrieve_documents_with_id(doc_ids, threshold=k)
        
        # If that fails, try a more generic approach
        generic_query = f"{role_type} skills experience"
        doc_ids = self.__retrieve_docs_id__(generic_query, k=k*2)
        return self.retrieve_documents_with_id(doc_ids, threshold=k)

    def _retrieve_by_skills(self, skills, k=5):
        """Retrieve resumes matching specific skills"""
        skills_text = " ".join(skills)
        skills_query = f"experience with {skills_text} professional skilled"
        
        # First try with all skills
        doc_ids = self.__retrieve_docs_id__(skills_query, k=k*2)
        if doc_ids:
            return self.retrieve_documents_with_id(doc_ids, threshold=k)
        
        # If that fails, try with individual skills
        results = []
        existing_ids = set()
        
        for skill in skills:
            skill_query = f"{skill} experience professional"
            skill_doc_ids = self.__retrieve_docs_id__(skill_query, k=max(3, k))
            
            # Add non-duplicate results
            docs = self.retrieve_documents_with_id(skill_doc_ids, threshold=3)
            for doc in docs:
                doc_id = self._extract_id_from_doc(doc)
                if doc_id and doc_id not in existing_ids:
                    results.append(doc)
                    existing_ids.add(doc_id)
            
            if len(results) >= k:
                break
        
        return results

    def _extract_id_from_doc(self, doc):
        """Extract the ID from a document string"""
        if not isinstance(doc, str):
            return None
            
        id_match = re.search(r'Applicant ID\s+(\d+)', doc)
        if id_match:
            return id_match.group(1)
        return None

    def _extract_ids_from_docs(self, docs):
        """Extract all IDs from a list of documents"""
        ids = set()
        for doc in docs:
            doc_id = self._extract_id_from_doc(doc)
            if doc_id:
                ids.add(doc_id)
        return ids

    def evaluate_resume_shortlisting(self, shortlisted_candidates, required_count):
      """
      Evaluate the quality of resume shortlisting with RAGAS-inspired metrics
      
      Args:
          shortlisted_candidates: Dictionary of shortlisted candidates with analysis
          required_count: Number of candidates requested
      """
      metrics = {}
      
      # Traditional metrics - keep existing ones
      
      # 1. Relevance score - average of overall scores
      if shortlisted_candidates:
          overall_scores = [candidate.get("overall_score", 0) for candidate in shortlisted_candidates.values()]
          metrics["relevance_score"] = sum(overall_scores) / len(overall_scores)
      else:
          metrics["relevance_score"] = 0
      
      # 2. Selection coverage - did we find enough candidates?
      metrics["selection_coverage"] = min(1.0, len(shortlisted_candidates) / required_count) * 100
      
      # 3. Skill diversity - how many unique skills are represented?
      all_skills = []
      for candidate in shortlisted_candidates.values():
          analysis = candidate.get("analysis", {})
          skills = analysis.get("skills", [])
          all_skills.extend(skills)
      
      unique_skills = len(set(all_skills))
      metrics["skill_diversity"] = min(100, unique_skills * 10)  # 10 points per unique skill, max 100
      
      # 4. Experience range - what's the range of experience years?
      if shortlisted_candidates:
          experience_years = [candidate.get("analysis", {}).get("years_experience", 0) 
                            for candidate in shortlisted_candidates.values()]
          
          if experience_years:
              metrics["min_experience"] = min(experience_years)
              metrics["max_experience"] = max(experience_years)
              metrics["avg_experience"] = sum(experience_years) / len(experience_years)
          else:
              metrics["min_experience"] = 0
              metrics["max_experience"] = 0
              metrics["avg_experience"] = 0
      else:
          metrics["min_experience"] = 0
          metrics["max_experience"] = 0
          metrics["avg_experience"] = 0
      
      # 5. Education distribution
      education_levels = {}
      for candidate in shortlisted_candidates.values():
          edu_level = candidate.get("analysis", {}).get("education_level", "unknown")
          education_levels[edu_level] = education_levels.get(edu_level, 0) + 1
          
      metrics["education_distribution"] = education_levels
      
      # RAGAS-inspired metrics
      
      # 1. Context Precision (RAGAS) - How precise are the retrieved candidates' skills?
      skill_precision = 0
      skill_count = 0
      for candidate in shortlisted_candidates.values():
          analysis = candidate.get("analysis", {})
          skill_score = analysis.get("skill_score", 0)
          if skill_score > 0:  # Only count if we have a valid score
              skill_precision += skill_score
              skill_count += 1
      
      metrics["context_precision"] = skill_precision / skill_count if skill_count > 0 else 0
      
      # 2. Context Recall (RAGAS) - How well do we cover the required skills?
      required_skills = set()
      for candidate in shortlisted_candidates.values():
          analysis = candidate.get("analysis", {})
          if "required_skills" in analysis:
              required_skills.update(analysis.get("required_skills", []))
      
      covered_skills = set()
      for candidate in shortlisted_candidates.values():
          analysis = candidate.get("analysis", {})
          covered_skills.update(analysis.get("skills", []))
      
      if required_skills:
          metrics["context_recall"] = len(covered_skills.intersection(required_skills)) / len(required_skills) * 100
      else:
          # If no required skills specified, assume good recall
          metrics["context_recall"] = 85
      
      # 3. Faithfulness (RAGAS) - How faithful are the candidates to the requirements?
      faithfulness_score = 0
      faithfulness_count = 0
      for candidate in shortlisted_candidates.values():
          analysis = candidate.get("analysis", {})
          experience_score = analysis.get("experience_score", 0)
          education_score = analysis.get("education_score", 0)
          
          if experience_score > 0 or education_score > 0:  # Only count if we have valid scores
              faithfulness_score += (experience_score + education_score) / 2
              faithfulness_count += 1
      
      metrics["faithfulness"] = faithfulness_score / faithfulness_count if faithfulness_count > 0 else 0
      
      # 4. Answer Relevance (RAGAS) - Overall relevance of the candidates
      answer_relevance = 0
      relevance_count = 0
      for candidate in shortlisted_candidates.values():
          overall_score = candidate.get("overall_score", 0)
          if overall_score > 0:  # Only count if we have a valid score
              answer_relevance += overall_score
              relevance_count += 1
      
      metrics["answer_relevance"] = answer_relevance / relevance_count if relevance_count > 0 else 0
      
      # Store metrics in metadata
      self.meta_data["evaluation_metrics"] = metrics
      
      return metrics