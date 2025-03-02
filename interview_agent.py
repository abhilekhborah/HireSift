# Enhanced interview_agent.py with improved parsing and invitation generation
import os
import json
import random
import datetime
import re
import logging
from typing import List, Dict, Optional, Union, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('interview_agent')

@dataclass
class InterviewSlot:
    """Data class for interview slots with enhanced validation"""
    candidate_id: str
    candidate_name: str
    interviewer_name: str
    date: str
    start_time: str
    end_time: str
    meeting_link: str
    email: str
    status: str = "pending"  # pending, scheduled, sent
    
    def __post_init__(self):
        """Validate the interview slot data"""
        # Ensure candidate_id is a string
        if not isinstance(self.candidate_id, str):
            self.candidate_id = str(self.candidate_id)
            
        # Ensure date format is valid
        try:
            if self.date.lower() in ["today", "tomorrow"]:
                pass  # Special cases handled elsewhere
            else:
                datetime.datetime.strptime(self.date, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format: {self.date}. Using tomorrow as fallback.")
            self.date = "tomorrow"
            
        # Validate time format
        for time_str in [self.start_time, self.end_time]:
            try:
                hours, minutes = map(int, time_str.split(':'))
                if not (0 <= hours < 24 and 0 <= minutes < 60):
                    raise ValueError
            except (ValueError, AttributeError):
                logger.warning(f"Invalid time format: {time_str}. Using default time.")
                self.start_time = "09:00"
                self.end_time = "09:45"
                break
        
        # Ensure candidate name is not "Unknown"
        if not self.candidate_name or self.candidate_name == "Unknown":
            self.candidate_name = f"Candidate {self.candidate_id}"


class InterviewScheduler:
    def __init__(self, interviewers=None):
        # Default interviewers if none provided
        self.interviewers = interviewers or [
            {"id": "1", "name": "Alice Chen", "email": "alice.chen@company.com", "expertise": ["Fullstack", "GenAI", "Frontend"], 
             "availability": [9, 10, 11, 14, 15, 16], "seniority": 4},
            {"id": "2", "name": "Bob Smith", "email": "bob.smith@company.com", "expertise": ["GenAI", "MLOps", "Machine Learning"], 
             "availability": [10, 11, 12, 13, 14], "seniority": 5},
            {"id": "3", "name": "Carol Wong", "email": "carol.wong@company.com", "expertise": ["Fullstack", "Frontend", "React"], 
             "availability": [9, 10, 13, 14, 15], "seniority": 3},
            {"id": "4", "name": "David Maguire", "email": "david.maguire@company.com", "expertise": ["Fullstack", "Backend", "Python"], 
             "availability": [11, 12, 13, 16, 17], "seniority": 4},
            {"id": "5", "name": "Eva Martinez", "email": "eva.martinez@company.com", "expertise": ["GenAI", "Data Science", "AI"], 
             "availability": [9, 10, 11, 14, 15], "seniority": 5}
        ]
        self.scheduled_interviews = []
        
        # Create skills taxonomy for matching related skills
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
        
        # Create reverse mapping for quick skill lookups
        self.skill_to_category = {}
        for category, skills in self.skills_taxonomy.items():
            for skill in skills:
                self.skill_to_category[skill.lower()] = category
                
        # Template storage for invitation emails
        self.email_templates = {
            "default": """
Subject: Interview Invitation

Dear {candidate_name},

We are pleased to invite you to an interview at our company. After reviewing your impressive qualifications and experience, we believe you would be a valuable addition to our team.

Interview Details:
- Date: {formatted_date}
- Time: {formatted_start} - {formatted_end}
- Format: Video Conference
- Meeting Link: {meeting_link}
- Interviewer: {interviewer_name}

During the interview, we'll discuss your technical experience, problem-solving approach, and how your skills align with our development needs. Please be prepared to discuss your past projects and specific experiences.

If you need to reschedule or have any questions prior to the interview, please reply to this email or contact our HR department at hr@company.com.

We look forward to speaking with you and learning more about your background and interests.

Best regards,
Recruiting Team
Siemens
Phone: (555) 123-4567
Email: recruiting@siemens.com
""",
#             "fullstack": """
# Subject: Interview Invitation: Full Stack Developer Position - {formatted_date}

# Dear {candidate_name},

# We are pleased to invite you to an interview for the Full Stack Developer position at our company. Your experience with {skills_list} caught our attention, and we're excited to discuss how your skills could contribute to our development team.

# Interview Details:
# - Date: {formatted_date}
# - Time: {formatted_start} - {formatted_end}
# - Format: Video Conference
# - Meeting Link: {meeting_link}
# - Interviewer: {interviewer_name}

# During the interview, we'll explore your experience with both front-end and back-end development, your approach to system architecture, and your problem-solving methodology. Please be prepared to discuss specific projects where you've worked across the entire stack.

# Technical Preparation:
# We recommend preparing examples of:
# - Front-end work (UI/UX design, JavaScript frameworks)
# - Back-end systems you've developed
# - Database design and optimization
# - API development and integration
# - Your experience with DevOps and deployment workflows

# If you need to reschedule or have any questions, please reply to this email or contact our HR department at hr@company.com.

# We look forward to speaking with you!

# Best regards,
# Recruiting Team
# Company Name
# Phone: (555) 123-4567
# Email: recruiting@company.com
# """,
#             "backend": """
# Subject: Interview Invitation: Backend Developer Position - {formatted_date}

# Dear {candidate_name},

# We are pleased to invite you to an interview for the Backend Developer position at our company. Your strong background in {skills_list} makes you an excellent candidate for our team.

# Interview Details:
# - Date: {formatted_date}
# - Time: {formatted_start} - {formatted_end}
# - Format: Video Conference
# - Meeting Link: {meeting_link}
# - Interviewer: {interviewer_name}

# During the interview, we'll focus on your experience with server-side architecture, database design, API development, and performance optimization. Please be prepared to discuss specific challenges you've solved in backend development.

# Technical Preparation:
# We recommend preparing examples of:
# - Server-side applications you've built
# - Database schema design and optimization
# - API architecture and documentation
# - Scalability solutions you've implemented
# - Security practices in your development workflow

# If you need to reschedule or have any questions, please reply to this email or contact our HR department at hr@company.com.

# We look forward to speaking with you!

# Best regards,
# Recruiting Team
# Company Name
# Phone: (555) 123-4567
# Email: recruiting@company.com
# """,
#             "frontend": """
# Subject: Interview Invitation: Frontend Developer Position - {formatted_date}

# Dear {candidate_name},

# We are pleased to invite you to an interview for the Frontend Developer position at our company. Your experience with {skills_list} aligns perfectly with what we're looking for in this role.

# Interview Details:
# - Date: {formatted_date}
# - Time: {formatted_start} - {formatted_end}
# - Format: Video Conference
# - Meeting Link: {meeting_link}
# - Interviewer: {interviewer_name}

# During the interview, we'll explore your experience with UI/UX design, JavaScript frameworks, responsive design, and frontend performance optimization. Please be prepared to discuss specific examples of your frontend work and approach to user interface development.

# Technical Preparation:
# We recommend preparing examples of:
# - UI components you've built
# - Responsive design implementation
# - State management approaches
# - Performance optimization techniques
# - Cross-browser compatibility solutions

# If you need to reschedule or have any questions, please reply to this email or contact our HR department at hr@company.com.

# We look forward to speaking with you!

# Best regards,
# Recruiting Team
# Company Name
# Phone: (555) 123-4567
# Email: recruiting@company.com
# """,
#             "data_scientist": """
# Subject: Interview Invitation: Data Scientist Position - {formatted_date}

# Dear {candidate_name},

# We are pleased to invite you to an interview for the Data Scientist position at our company. Your strong background in {skills_list} makes you an excellent candidate for our data science team.

# Interview Details:
# - Date: {formatted_date}
# - Time: {formatted_start} - {formatted_end}
# - Format: Video Conference
# - Meeting Link: {meeting_link}
# - Interviewer: {interviewer_name}

# During the interview, we'll explore your experience with statistical analysis, machine learning models, data visualization, and your approach to solving real-world data problems. Please be prepared to discuss specific projects where you've applied data science techniques.

# Technical Preparation:
# We recommend preparing examples of:
# - Data analysis projects you've completed
# - Machine learning models you've developed
# - Feature engineering approaches
# - Model evaluation and validation techniques
# - Data visualization methods you use to communicate findings

# If you need to reschedule or have any questions, please reply to this email or contact our HR department at hr@company.com.

# We look forward to speaking with you!

# Best regards,
# Recruiting Team
# Company Name
# Phone: (555) 123-4567
# Email: recruiting@company.com
# """,
#             "genai": """
# Subject: Interview Invitation: AI Engineer Position - {formatted_date}

# Dear {candidate_name},

# We are pleased to invite you to an interview for the AI Engineer position at our company. Your experience with {skills_list} aligns perfectly with our work in generative AI and machine learning systems.

# Interview Details:
# - Date: {formatted_date}
# - Time: {formatted_start} - {formatted_end}
# - Format: Video Conference
# - Meeting Link: {meeting_link}
# - Interviewer: {interviewer_name}

# During the interview, we'll explore your experience with AI model development, prompt engineering, fine-tuning, and AI application integration. Please be prepared to discuss specific AI projects you've worked on and your approach to developing AI solutions.

# Technical Preparation:
# We recommend preparing examples of:
# - AI models you've developed or fine-tuned
# - LLM integration work you've completed
# - Prompt engineering techniques you employ
# - Evaluation metrics for AI systems
# - Ethical considerations in your AI development

# If you need to reschedule or have any questions, please reply to this email or contact our HR department at hr@company.com.

# We look forward to speaking with you!

# Best regards,
# Recruiting Team
# Company Name
# Phone: (555) 123-4567
# Email: recruiting@company.com
# """,
#             "business_analyst": """
# Subject: Interview Invitation: Business Analyst Position - {formatted_date}

# Dear {candidate_name},

# We are pleased to invite you to an interview for the Business Analyst position at our company. Your strong background in {skills_list} makes you an excellent candidate for our team.

# Interview Details:
# - Date: {formatted_date}
# - Time: {formatted_start} - {formatted_end}
# - Format: Video Conference
# - Meeting Link: {meeting_link}
# - Interviewer: {interviewer_name}

# During the interview, we'll explore your experience with requirements gathering, process analysis, documentation, and stakeholder management. Please be prepared to discuss specific projects where you've successfully translated business needs into technical solutions.

# Preparation:
# We recommend preparing examples of:
# - Requirements documents you've created
# - Business processes you've analyzed and improved
# - Project management methodologies you're familiar with
# - Tools you use for documentation and analysis
# - Your approach to stakeholder communication

# If you need to reschedule or have any questions, please reply to this email or contact our HR department at hr@company.com.

# We look forward to speaking with you!

# Best regards,
# Recruiting Team
# Company Name
# Phone: (555) 123-4567
# Email: recruiting@company.com
# """
        }
    
    def generate_meeting_link(self):
        """Generate a mock Google Meet link with proper format"""
        # Ensure we're generating a valid-looking meeting link
        valid_chars = 'abcdefghijklmnopqrstuvwxyz'
        meeting_id = ''.join(random.choices(valid_chars, k=3)) + '-' + \
                     ''.join(random.choices(valid_chars, k=4)) + '-' + \
                     ''.join(random.choices(valid_chars, k=3))
        return f"https://meet.google.com/{meeting_id}"
    
    def calculate_expertise_match_score(self, candidate_skills: List[str], interviewer_expertise: List[str]) -> float:
        """
        Calculate expertise match score between candidate skills and interviewer expertise
        with consideration for related skills
        
        Args:
            candidate_skills: List of candidate's skills
            interviewer_expertise: List of interviewer's expertise areas
            
        Returns:
            Match score from 0-100
        """
        if not candidate_skills or not interviewer_expertise:
            return 0
            
        # Normalize all skills to lowercase
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        interviewer_expertise_lower = [e.lower() for e in interviewer_expertise]
        
        # Calculate direct matches
        direct_matches = sum(1 for skill in candidate_skills_lower if skill in interviewer_expertise_lower)
        
        # Calculate related skill matches
        related_matches = 0
        for skill in candidate_skills_lower:
            if skill not in interviewer_expertise_lower:  # Only check if not direct match
                skill_category = self.skill_to_category.get(skill)
                if skill_category:
                    # Check if interviewer has expertise in related skills
                    for expertise in interviewer_expertise_lower:
                        if self.skill_to_category.get(expertise) == skill_category:
                            related_matches += 0.5  # Count as half a match
                            break
        
        # Calculate total matches
        total_matches = direct_matches + related_matches
        
        # Calculate match percentage
        match_percentage = (total_matches / len(candidate_skills)) * 100
        
        # Add bonus for higher coverage of candidate skills
        match_percentage = min(match_percentage * 1.2, 100)
        
        return match_percentage
    
    def find_best_interviewer(self, candidate_expertise):
        """Find the best interviewer based on expertise match with better handling of edge cases"""
        best_match = None
        best_score = -1
        
        # Handle empty or invalid expertise
        if not candidate_expertise:
            logger.warning("Empty candidate expertise provided, using default interviewer")
            return self.interviewers[0] if self.interviewers else None
            
        # Normalize expertise to list if needed
        if isinstance(candidate_expertise, str):
            candidate_expertise = [candidate_expertise]
            
        # Track current interview load for workload balancing
        interviewer_workloads = {}
        for interview in self.scheduled_interviews:
            interviewer_name = interview.interviewer_name
            interviewer_workloads[interviewer_name] = interviewer_workloads.get(interviewer_name, 0) + 1
        
        for interviewer in self.interviewers:
            interviewer_expertise = interviewer.get("expertise", [])
            
            # Handle case where expertise might be a string
            if isinstance(interviewer_expertise, str):
                interviewer_expertise = [interviewer_expertise]
            
            # Calculate expertise match score using the enhanced algorithm
            expertise_match = self.calculate_expertise_match_score(
                candidate_expertise, 
                interviewer_expertise
            )
            
            # Calculate a workload penalty for balancing
            current_load = interviewer_workloads.get(interviewer.get("name", ""), 0)
            workload_factor = max(0, 1.0 - (current_load * 0.1))  # Decrease score as workload increases
            
            # Calculate total score with both expertise match and workload considerations
            total_score = expertise_match * workload_factor
            
            if total_score > best_score:
                best_score = total_score
                best_match = interviewer
                
        # If no match found, return random interviewer as fallback
        if best_match is None and self.interviewers:
            logger.warning("No matching interviewer found, using random interviewer")
            return random.choice(self.interviewers)
            
        # If still no match, create a default interviewer
        if best_match is None:
            logger.warning("No interviewers available, creating default interviewer")
            default_interviewer = {
                "id": "default",
                "name": "Default Interviewer",
                "email": "default@company.com",
                "expertise": ["General"],
                "availability": list(range(9, 17))  # 9am to 5pm
            }
            self.interviewers.append(default_interviewer)
            return default_interviewer
            
        return best_match
    
    def parse_time_request(self, time_str):
        """Parse time string like '9am' or '2:30pm' with enhanced robustness"""
        if not time_str:
            return 9, 0  # Default to 9am
            
        try:
            time_str = time_str.lower().strip()
            
            # Handle formats like "9am", "9:00am", "14:00", "2pm", etc.
            if "am" in time_str or "pm" in time_str:
                # Strip am/pm for processing
                is_pm = "pm" in time_str
                time_value = time_str.replace("am", "").replace("pm", "").strip()
                
                if ":" in time_value:
                    hour, minute = map(int, time_value.split(":"))
                else:
                    hour = int(time_value)
                    minute = 0
                    
                if is_pm and hour < 12:
                    hour += 12
                elif hour == 12 and not is_pm:
                    hour = 0  # 12am = 0 hours
            else:
                # Assume 24-hour format
                if ":" in time_str:
                    hour, minute = map(int, time_str.split(":"))
                else:
                    hour = int(time_str)
                    minute = 0
                    
            # Validate hour and minute
            if hour < 0 or hour > 23:
                logger.warning(f"Invalid hour {hour}, using 9am as default")
                hour = 9  # Default to 9am if invalid hour
                
            if minute < 0 or minute > 59:
                logger.warning(f"Invalid minute {minute}, using 0 as default")
                minute = 0  # Default to top of hour if invalid minute
                
            return hour, minute
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing time '{time_str}': {e}, using 9am as default")
            # If parsing fails, return default time 9am
            return 9, 0
    
    def schedule_interview(self, candidate_id, candidate_name, candidate_expertise, 
                        requested_time=None, requested_date=None, interviewer=None):
        """Schedule an interview with the specified parameters with improved error handling"""
        # Process candidate information
        candidate_id = str(candidate_id) if candidate_id else str(random.randint(1000, 9999))
        candidate_name = candidate_name or f"Candidate {candidate_id}"
        
        # Normalize expertise to a list
        if isinstance(candidate_expertise, str):
            candidate_expertise = [candidate_expertise]
        elif not candidate_expertise:
            candidate_expertise = ["General"]
            
        # Process the date request using improved date parsing
        if isinstance(requested_date, str):
            # Parse using the new date parser
            processed_date = self.parse_date_request(requested_date)
        else:
            # If date is None or not a string, use tomorrow
            today = datetime.datetime.now()
            days_ahead = 1 if today.weekday() < 4 else 3  # Skip to Monday if today is Friday
            processed_date = (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Find best interviewer if not specified
        interviewer_obj = None
        if interviewer:
            # Try to find the specified interviewer
            interviewer_obj = next((i for i in self.interviewers if i["name"].lower() == interviewer.lower()), None)
            
        if not interviewer_obj:
            # Calculate expertise match scores for all interviewers
            interviewer_scores = []
            for i, interviewer in enumerate(self.interviewers):
                interviewer_expertise = interviewer.get("expertise", [])
                if isinstance(interviewer_expertise, str):
                    interviewer_expertise = [interviewer_expertise]
                    
                # Calculate match score based on overlapping skills
                expertise_match = self.calculate_expertise_match_score(candidate_expertise, interviewer_expertise)
                
                # Calculate current workload for this interviewer
                scheduled_count = sum(1 for interview in self.scheduled_interviews 
                                    if interview.interviewer_name == interviewer["name"])
                
                # Calculate workload factor (decreases as workload increases)
                workload_factor = max(0, 1.0 - (scheduled_count * 0.15))
                
                # Calculate final score
                final_score = expertise_match * workload_factor
                interviewer_scores.append((interviewer, final_score, expertise_match))
            
            # Sort by score (highest first) and select best match
            if interviewer_scores:
                interviewer_scores.sort(key=lambda x: x[1], reverse=True)
                interviewer_obj = interviewer_scores[0][0]
                
                # Log the expertise match for the selected interviewer
                logger.info(f"Selected interviewer {interviewer_obj['name']} with expertise match: {interviewer_scores[0][2]:.1f}%")
            else:
                interviewer_obj = self.find_best_interviewer(candidate_expertise)
            
        if not interviewer_obj:
            logger.error("Could not find or create an interviewer")
            # Create a default interviewer as fallback
            interviewer_obj = {
                "id": "default",
                "name": "Default Interviewer",
                "expertise": ["General"],
                "availability": list(range(9, 17)),
                "email": "interviewer@company.com"
            }
        
        # Get already scheduled times for this interviewer on this date
        booked_times = []
        for interview in self.scheduled_interviews:
            if interview.interviewer_name == interviewer_obj["name"] and interview.date == processed_date:
                # Convert start and end times to hours for easier comparison
                try:
                    start_hour, start_min = map(int, interview.start_time.split(':'))
                    end_hour, end_min = map(int, interview.end_time.split(':'))
                    
                    # Mark all hours between start and end as booked (including partial hours)
                    start_decimal = start_hour + (start_min / 60)
                    end_decimal = end_hour + (end_min / 60)
                    
                    # Add a buffer of 15 minutes before and after
                    start_with_buffer = max(0, start_decimal - 0.25)
                    end_with_buffer = min(24, end_decimal + 0.25)
                    
                    booked_times.append((start_with_buffer, end_with_buffer))
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing interview times: {e}")
                    continue
        
        # Parse requested time or find first available slot
        hour = None
        minute = None
        
        if requested_time:
            # Parse the requested time
            hour, minute = self.parse_time_request(requested_time)
            requested_decimal = hour + (minute / 60)
            
            # Check if the time is already booked
            is_booked = any(start <= requested_decimal < end for start, end in booked_times)
            
            # Check if interviewer is available at this time and time isn't booked
            interviewer_availability = interviewer_obj.get("availability", list(range(9, 17)))
            if hour not in interviewer_availability or is_booked:
                logger.info(f"Requested time {hour}:{minute} is not available, finding alternative")
                # Find closest available time that isn't booked
                available_hours = sorted(interviewer_availability)
                available_slots = []
                
                # Check all possible slots with this interviewer
                for h in available_hours:
                    for m in [0, 15, 30, 45]:
                        slot_start = h + (m / 60)
                        slot_end = slot_start + 0.75  # 45 minutes
                        
                        if not any(start <= slot_start < end or start < slot_end <= end or 
                                (slot_start <= start and slot_end >= end) for start, end in booked_times):
                            available_slots.append((h, m))
                
                if available_slots:
                    # Choose the slot closest to requested time
                    closest_slot = min(available_slots, 
                                    key=lambda x: abs((x[0] + (x[1]/60)) - requested_decimal))
                    hour, minute = closest_slot
                    logger.info(f"Found alternative time slot: {hour}:{minute}")
                else:
                    # Try other interviewers if no slots available with current interviewer
                    logger.warning(f"No available slots for {interviewer_obj['name']}, trying other interviewers")
                    
                    # Find alternative interviewers with expertise match
                    alternative_interviewers = []
                    for other_interviewer in self.interviewers:
                        if other_interviewer != interviewer_obj:
                            # Check if this interviewer has expertise match
                            other_expertise = other_interviewer.get("expertise", [])
                            if isinstance(other_expertise, str):
                                other_expertise = [other_expertise]
                                
                            # Calculate expertise match score
                            match_score = self.calculate_expertise_match_score(candidate_expertise, other_expertise)
                            
                            if match_score > 0:
                                # Get this interviewer's available slots
                                other_availability = other_interviewer.get("availability", [])
                                other_booked_times = []
                                
                                for interview in self.scheduled_interviews:
                                    if interview.interviewer_name == other_interviewer["name"] and interview.date == processed_date:
                                        try:
                                            start_hour, start_min = map(int, interview.start_time.split(':'))
                                            end_hour, end_min = map(int, interview.end_time.split(':'))
                                            
                                            start_decimal = start_hour + (start_min / 60)
                                            end_decimal = end_hour + (end_min / 60)
                                            
                                            other_booked_times.append((start_decimal, end_decimal))
                                        except (ValueError, AttributeError):
                                            continue
                                
                                # Find available slots with this interviewer
                                interviewer_slots = []
                                for h in sorted(other_availability):
                                    for m in [0, 15, 30, 45]:
                                        slot_start = h + (m / 60)
                                        slot_end = slot_start + 0.75  # 45 minutes
                                        
                                        if not any(start <= slot_start < end or start < slot_end <= end or 
                                                (slot_start <= start and slot_end >= end) for start, end in other_booked_times):
                                            # Calculate how close this slot is to requested time
                                            time_diff = abs((h + (m/60)) - requested_decimal)
                                            interviewer_slots.append((h, m, time_diff))
                                
                                if interviewer_slots:
                                    # Sort by closeness to requested time
                                    interviewer_slots.sort(key=lambda x: x[2])
                                    best_slot = interviewer_slots[0]
                                    alternative_interviewers.append({
                                        "interviewer": other_interviewer,
                                        "slot": (best_slot[0], best_slot[1]),
                                        "score": match_score,
                                        "time_diff": best_slot[2]
                                    })
                    
                    if alternative_interviewers:
                        # Sort by expertise score (descending) and time difference (ascending)
                        alternative_interviewers.sort(key=lambda x: (-x["score"], x["time_diff"]))
                        best_alternative = alternative_interviewers[0]
                        
                        interviewer_obj = best_alternative["interviewer"]
                        hour, minute = best_alternative["slot"]
                        logger.info(f"Using alternative interviewer: {interviewer_obj['name']} at {hour}:{minute}")
                    else:
                        # Last resort: try next day with original interviewer
                        try:
                            tomorrow = datetime.datetime.strptime(processed_date, "%Y-%m-%d") + datetime.timedelta(days=1)
                            processed_date = tomorrow.strftime("%Y-%m-%d")
                            hour = interviewer_availability[0] if interviewer_availability else 9
                            minute = 0
                            logger.info(f"Scheduling for next day: {processed_date} at {hour}:{minute}")
                        except Exception as e:
                            logger.error(f"Error calculating next day: {e}")
                            # Fallback to default
                            hour = 9
                            minute = 0
        else:
            # No specific time requested, use first available time slot that isn't booked
            available_found = False
            interviewer_availability = interviewer_obj.get("availability", list(range(9, 17)))
            
            # Prioritize morning slots (9-12) then afternoon (1-5)
            preferred_hours = [9, 10, 11, 13, 14, 15, 16]
            for h in sorted(interviewer_availability, key=lambda x: (0 if x in preferred_hours else 1, x)):
                for m in [0, 15, 30, 45]:
                    slot_start = h + (m / 60)
                    slot_end = slot_start + 0.75  # 45 minutes
                    
                    if not any(start <= slot_start < end or start < slot_end <= end or 
                            (slot_start <= start and slot_end >= end) for start, end in booked_times):
                        hour = h
                        minute = m
                        available_found = True
                        break
                if available_found:
                    break
            
            if not available_found:
                logger.warning(f"No available time slots for {processed_date}, trying next day")
                # Try the next day
                try:
                    tomorrow = datetime.datetime.strptime(processed_date, "%Y-%m-%d") + datetime.timedelta(days=1)
                    processed_date = tomorrow.strftime("%Y-%m-%d")
                    hour = interviewer_availability[0] if interviewer_availability else 9
                    minute = 0
                except Exception as e:
                    logger.error(f"Error calculating next day: {e}")
                    # Fallback to default
                    hour = 9
                    minute = 0
        
        # Format times for display
        start_time = f"{hour:02d}:{minute:02d}"
        end_hour = hour + ((minute + 45) // 60)
        end_minute = (minute + 45) % 60
        end_time = f"{end_hour:02d}:{end_minute:02d}"
        
        # Create interview slot
        meeting_link = self.generate_meeting_link()
        try:
            interview = InterviewSlot(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                interviewer_name=interviewer_obj["name"],
                date=processed_date,
                start_time=start_time,
                end_time=end_time,
                meeting_link=meeting_link,
                email=interviewer_obj.get("email", "interviewer@company.com"),
                status="scheduled"
            )
            
            # Add to scheduled interviews list
            self.scheduled_interviews.append(interview)
            logger.info(f"Successfully scheduled interview for {candidate_name} with {interviewer_obj['name']} on {processed_date} at {start_time}")
            return interview
        except Exception as e:
            logger.error(f"Error creating interview slot: {e}")
            return None
    
    def get_scheduled_interviews(self):
        """Return all scheduled interviews"""
        return self.scheduled_interviews
    
    def generate_invitation_draft(self, interview):
        """
        Generate a personalized interview invitation draft for a specific candidate
        
        Args:
            interview: Dictionary or InterviewSlot with interview details
            
        Returns:
            String containing the email draft
        """
        try:
            if isinstance(interview, dict):
                # Handle dictionary format
                candidate_name = interview.get("candidate_name", "Candidate")
                date = interview.get("date", interview.get("interview_date", ""))
                start_time = interview.get("start_time", "")
                end_time = interview.get("end_time", "")
                meeting_link = interview.get("meeting_link", "")
                interviewer_name = interview.get("interviewer_name", "")
                candidate_skills = interview.get("candidate_key_skills", [])
            else:
                # Handle InterviewSlot format
                candidate_name = interview.candidate_name
                date = interview.date
                start_time = interview.start_time
                end_time = interview.end_time
                meeting_link = interview.meeting_link
                interviewer_name = interview.interviewer_name
                candidate_skills = []
            
            # Ensure we have valid values for all fields
            candidate_name = candidate_name if candidate_name and candidate_name != "Unknown" else "Candidate"
            date = date or "tomorrow"
            start_time = start_time or "09:00"
            end_time = end_time or "09:45"
            meeting_link = meeting_link or "https://meet.google.com/abc-defg-hij"
            interviewer_name = interviewer_name or "Interviewer"
            
            # Format the date for readability if it's in ISO format
            formatted_date = date
            try:
                if date and date.count("-") == 2:  # Looks like ISO format (YYYY-MM-DD)
                    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
                    formatted_date = date_obj.strftime("%A, %B %d, %Y")  # e.g., "Thursday, July 4, 2024"
            except Exception as date_e:
                print(f"Date formatting error: {date_e}")
                formatted_date = date
                
            # Format times for readability
            formatted_start = start_time
            formatted_end = end_time
            try:
                if ":" in start_time:
                    hours, minutes = map(int, start_time.split(":"))
                    am_pm = "AM" if hours < 12 else "PM"
                    display_hour = hours if hours <= 12 else hours - 12
                    display_hour = 12 if display_hour == 0 else display_hour
                    formatted_start = f"{display_hour}:{minutes:02d} {am_pm}"
                    
                if ":" in end_time:
                    hours, minutes = map(int, end_time.split(":"))
                    am_pm = "AM" if hours < 12 else "PM"
                    display_hour = hours if hours <= 12 else hours - 12
                    display_hour = 12 if display_hour == 0 else display_hour
                    formatted_end = f"{display_hour}:{minutes:02d} {am_pm}"
            except Exception as time_e:
                print(f"Time formatting error: {time_e}")
                formatted_start = start_time
                formatted_end = end_time
            
            # Determine role type based on candidate skills or interviewer expertise
            role_type = "Software Developer"
            if candidate_skills:
                if any(skill.lower() in ["genai", "machine learning", "ai", "nlp", "pytorch", "tensorflow"] 
                       for skill in candidate_skills):
                    role_type = "AI Engineer"
                    template_key = "genai"
                elif any(skill.lower() in ["frontend", "react", "angular", "vue", "html", "css"] 
                         for skill in candidate_skills):
                    role_type = "Frontend Developer"
                    template_key = "frontend"
                elif any(skill.lower() in ["backend", "django", "flask", "spring", "express"] 
                         for skill in candidate_skills):
                    role_type = "Backend Developer"
                    template_key = "backend"
                elif any(skill.lower() in ["business analyst", "scrum master", "jira", "agile"] 
                         for skill in candidate_skills):
                    role_type = "Business Analyst"
                    template_key = "business_analyst"
                elif any(skill.lower() in ["data scientist", "statistics", "pandas", "numpy", "data analysis"] 
                         for skill in candidate_skills):
                    role_type = "Data Scientist"
                    template_key = "data_scientist"
                else:
                    role_type = "Full Stack Developer"
                    template_key = "fullstack"
            else:
                # Try to determine from interviewer expertise
                if interviewer_name:
                    interviewer = next((i for i in self.interviewers if i["name"] == interviewer_name), None)
                    if interviewer:
                        expertise = interviewer.get("expertise", [])
                        if any(skill.lower() in ["genai", "ai", "machine learning"] for skill in expertise):
                            role_type = "AI Engineer"
                            template_key = "genai"
                        elif any(skill.lower() in ["frontend"] for skill in expertise):
                            role_type = "Frontend Developer"
                            template_key = "frontend"
                        elif any(skill.lower() in ["backend"] for skill in expertise):
                            role_type = "Backend Developer"
                            template_key = "backend"
                        else:
                            role_type = "Full Stack Developer"
                            template_key = "fullstack"
                    else:
                        template_key = "default"
                else:
                    template_key = "default"
            
            # Create a personalized skills section
            skills_section = ""
            skills_list = ""
            if candidate_skills:
                skills_list = ", ".join(candidate_skills[:3])
                if len(candidate_skills) > 3:
                    skills_list += ", and other relevant technologies"
                skills_section = f"\nBased on your experience with {skills_list}, we're excited to discuss how your skills align with our team's needs."
            
            # Get the appropriate template
            template = self.email_templates.get(template_key, self.email_templates["default"])
            
            # Fill in the template
            email_draft = template.format(
                candidate_name=candidate_name,
                formatted_date=formatted_date,
                formatted_start=formatted_start,
                formatted_end=formatted_end,
                meeting_link=meeting_link,
                interviewer_name=interviewer_name,
                role_type=role_type,
                skills_section=skills_section,
                skills_list=skills_list
            )
                
            return email_draft.strip()
        except Exception as e:
            print(f"Error generating email draft: {e}")
            # Return a basic fallback template that will always work
            return f"""
    Subject: Interview Scheduled for Software Developer Position

    Dear {candidate_name},

    We're pleased to invite you to an interview for the Software Developer position at our company.

    The interview is scheduled for {date} at {start_time}.

    Please contact us if you need to reschedule.

    Best regards,
    Recruiting Team
            """.strip()

    def parse_date_request(self, requested_date):
        """
        Parse date from various formats, with enhanced handling for day names
        
        Args:
            requested_date: Date string in various formats ("tomorrow", "Thursday", "2024-07-04", etc.)
            
        Returns:
            Formatted date string in YYYY-MM-DD format
        """
        if not requested_date:
            # Default to tomorrow
            today = datetime.datetime.now()
            days_ahead = 1 if today.weekday() < 4 else 3  # Skip to Monday if today is Friday
            return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Convert to lowercase for case-insensitive comparison
        requested_date_lower = requested_date.lower()
        
        # Handle "today" and "tomorrow"
        if requested_date_lower == "today":
            return datetime.datetime.now().strftime("%Y-%m-%d")
        elif requested_date_lower == "tomorrow":
            today = datetime.datetime.now()
            days_ahead = 1 if today.weekday() < 4 else 3  # Skip to Monday if today is Friday
            return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif requested_date_lower == "next week":
            # Schedule for Monday of next week
            today = datetime.datetime.now()
            days_ahead = 7 - today.weekday()  # Days until Sunday
            days_ahead += 1  # Add one more day to get to Monday
            return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Handle day names (monday, tuesday, etc.)
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
            "friday": 4, "saturday": 5, "sunday": 6
        }
        
        # Check for day names with prefixes like "next Monday", "this Thursday"
        for day_name in day_map.keys():
            if day_name in requested_date_lower:
                today = datetime.datetime.now()
                target_day = day_map[day_name]
                
                # Calculate days ahead based on modifiers
                if "next" in requested_date_lower:
                    # "Next X" means the X after the upcoming one
                    days_ahead = (target_day - today.weekday()) % 7
                    if days_ahead == 0:  # If today is the target day
                        days_ahead = 7  # Go to next week
                    days_ahead += 7  # Add another week for "next"
                else:
                    # "This X" or just "X" means the upcoming X
                    days_ahead = (target_day - today.weekday()) % 7
                    if days_ahead == 0:  # If today is the target day
                        if "this" in requested_date_lower:
                            days_ahead = 0  # Keep it today
                        else:
                            days_ahead = 7  # Go to next week
                            
                return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # If we reach here, check if it's just a direct day name match
        if requested_date_lower in day_map:
            today = datetime.datetime.now()
            target_day = day_map[requested_date_lower]
            days_ahead = (target_day - today.weekday()) % 7
            
            # If the day is today, schedule for next week instead
            if days_ahead == 0:
                days_ahead = 7
                
            return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Handle date formats like mm/dd/yyyy or yyyy-mm-dd
        try:
            # Try various formats
            date_formats = [
                "%Y-%m-%d",  # 2024-07-04
                "%m/%d/%Y",  # 07/04/2024
                "%m-%d-%Y",  # 07-04-2024
                "%m/%d/%y",  # 07/04/24
                "%m-%d-%y",  # 07-04-24
                "%d/%m/%Y",  # 04/07/2024 (European format)
                "%d-%m-%Y",  # 04-07-2024 (European format)
                "%B %d, %Y", # July 4, 2024
                "%b %d, %Y"  # Jul 4, 2024
            ]
            
            for date_format in date_formats:
                try:
                    parsed_date = datetime.datetime.strptime(requested_date, date_format)
                    # Check if the date is in the past
                    if parsed_date.date() < datetime.datetime.now().date():
                        # If it's in the past, assume it's for next year
                        if "%Y" in date_format or "%y" in date_format:
                            # Only adjust if year was explicitly specified
                            logger.warning(f"Date {requested_date} is in the past, scheduling for tomorrow instead")
                            today = datetime.datetime.now()
                            days_ahead = 1 if today.weekday() < 4 else 3
                            return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                    
                    return parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            
            # If no format worked, raise exception to use fallback
            raise ValueError(f"Could not parse date format: {requested_date}")
        except Exception as e:
            logger.warning(f"Error parsing date: {e}, using tomorrow")
            # Fallback to tomorrow
            today = datetime.datetime.now()
            days_ahead = 1 if today.weekday() < 4 else 3
            return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    def schedule_multiple_interviews(self, candidates, requested_times):
        """
        Schedule multiple interviews for different candidates at specified times
        
        Args:
            candidates: List of candidate information dictionaries with ID, name, and expertise
            requested_times: List of requested time strings (e.g., "9am", "11am")
            
        Returns:
            List of scheduled interview slots
        """
        scheduled = []
        interviewer_allocations = {}  # Track interviewer allocations by time slot
        
        # Handle empty candidates case
        if not candidates:
            logger.warning("No candidates provided for scheduling")
            return []
            
        # Ensure candidates is a list
        if not isinstance(candidates, list):
            candidates = [candidates]
            
        # Handle case where requested_times is a string
        if isinstance(requested_times, str):
            requested_times = [requested_times]
            
        # If no times provided, use defaults
        if not requested_times:
            requested_times = ["9am"]
        
        # Validate input lists have the same length
        if len(candidates) > len(requested_times):
            # Extend requested_times with the last time
            last_time = requested_times[-1] if requested_times else "9am"
            requested_times.extend([last_time] * (len(candidates) - len(requested_times)))
        elif len(candidates) < len(requested_times):
            # Truncate requested_times to match candidates
            requested_times = requested_times[:len(candidates)]
        
        # Sort by requested time to handle earlier times first
        scheduling_requests = list(zip(candidates, requested_times))
        
        # Process each candidate
        for candidate, requested_time in scheduling_requests:
            # Ensure candidate is a dictionary
            if isinstance(candidate, str):
                # Try to parse the string into a candidate dictionary
                try:
                    # Try to import the ChatBot for parsing
                    try:
                        from llm_agent import ChatBot
                        temp_bot = ChatBot(api_key=os.getenv("GROQ_API_KEY"), model="gemini-2.0-flash")
                        candidate = temp_bot._parse_candidate_from_string(candidate)
                    except ImportError:
                        # Basic fallback parsing if LLM agent not available
                        candidate = self._parse_candidate_from_string(candidate)
                except Exception as e:
                    logger.error(f"Error parsing candidate: {e}")
                    # Create a minimal candidate dict
                    candidate_id = str(random.randint(1000, 9999))
                    candidate = {
                        "ID": candidate_id,
                        "Name": f"Candidate {candidate_id}",
                        "Skills": ["General"]
                    }
            
            candidate_id = str(candidate.get("ID", ""))
            candidate_name = candidate.get("Name", f"Candidate {candidate_id}")
            candidate_expertise = candidate.get("Skills", ["General"])
            
            # Parse the requested time
            try:
                hour, minute = self.parse_time_request(requested_time)
                time_key = f"{hour:02d}:{minute:02d}"
            except Exception as e:
                logger.error(f"Error parsing time {requested_time}: {e}")
                # Use default time if parsing fails
                hour, minute = 9, 0
                time_key = "09:00"
            
            # Check if we already have an interviewer allocated at this time
            if time_key in interviewer_allocations:
                logger.info(f"Time slot {time_key} already allocated, finding alternative")
                # Find a different interviewer with matching skills
                used_interviewer = interviewer_allocations[time_key]
                available_interviewers = [
                    i for i in self.interviewers 
                    if i["name"] != used_interviewer["name"] and 
                    hour in i.get("availability", []) and
                    any(skill in i.get("expertise", []) for skill in candidate_expertise)
                ]
                
                if available_interviewers:
                    # Use the best matching available interviewer
                    interviewer = max(
                        available_interviewers,
                        key=lambda i: self.calculate_expertise_match_score(candidate_expertise, i.get("expertise", []))
                    )
                    logger.info(f"Found alternative interviewer: {interviewer['name']}")
                else:
                    logger.info(f"No alternative interviewer, finding next available slot")
                    # No other interviewer available at this time, find next available slot
                    next_slot = self.find_next_available_slot(hour, minute, candidate_expertise)
                    if next_slot:
                        hour, minute = next_slot["hour"], next_slot["minute"]
                        time_key = f"{hour:02d}:{minute:02d}"
                        interviewer = next_slot["interviewer"]
                        logger.info(f"Found next available slot: {hour}:{minute} with {interviewer['name']}")
                    else:
                        # Fallback: move to next day
                        logger.warning(f"No available slots found, defaulting to first interviewer")
                        hour = self.interviewers[0].get("availability", [9])[0]
                        minute = 0
                        interviewer = self.find_best_interviewer(candidate_expertise)
            else:
                # Find best interviewer with availability at this time
                available_interviewers = [
                    i for i in self.interviewers 
                    if hour in i.get("availability", []) and
                    any(skill in i.get("expertise", []) for skill in candidate_expertise)
                ]
                
                if available_interviewers:
                    # Sort by expertise match score to find the best match
                    scored_interviewers = [
                        (i, self.calculate_expertise_match_score(candidate_expertise, i.get("expertise", [])))
                        for i in available_interviewers
                    ]
                    sorted_interviewers = sorted(scored_interviewers, key=lambda x: x[1], reverse=True)
                    
                    # Use the top matching interviewer
                    interviewer = sorted_interviewers[0][0]
                    match_score = sorted_interviewers[0][1]
                    logger.info(f"Matched interviewer {interviewer['name']} for time {hour}:{minute} with match score {match_score:.1f}%")
                else:
                    logger.info(f"No interviewer available at {hour}:{minute}, finding alternative")
                    # No interviewer available, find closest available time
                    next_slot = self.find_next_available_slot(hour, minute, candidate_expertise)
                    if next_slot:
                        hour, minute = next_slot["hour"], next_slot["minute"]
                        time_key = f"{hour:02d}:{minute:02d}"
                        interviewer = next_slot["interviewer"]
                        logger.info(f"Found alternative slot: {hour}:{minute} with {interviewer['name']}")
                    else:
                        # Fallback to first available interviewer
                        logger.warning(f"No available slots found, defaulting to best matching interviewer")
                        interviewer = self.find_best_interviewer(candidate_expertise)
            
            # Save allocation for this time slot
            interviewer_allocations[time_key] = interviewer
            
            # Schedule the interview
            today = datetime.datetime.now()
            requested_date = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Format times for display
            start_time = f"{hour:02d}:{minute:02d}"
            end_hour = hour + ((minute + 45) // 60)
            end_minute = (minute + 45) % 60
            end_time = f"{end_hour:02d}:{end_minute:02d}"
            
            # Create interview slot
            meeting_link = self.generate_meeting_link()
            try:
                interview = InterviewSlot(
                    candidate_id=candidate_id,
                    candidate_name=candidate_name,
                    interviewer_name=interviewer["name"],
                    date=requested_date,
                    start_time=start_time,
                    end_time=end_time,
                    meeting_link=meeting_link,
                    email=interviewer.get("email", "interviewer@company.com"),
                    status="scheduled"
                )
                
                self.scheduled_interviews.append(interview)
                scheduled.append(interview)
                logger.info(f"Scheduled interview for {candidate_name} with {interviewer['name']} at {start_time}")
            except Exception as e:
                logger.error(f"Error creating interview slot: {e}")
        
        return scheduled

    def find_next_available_slot(self, after_hour, after_minute, candidate_expertise):
        """Find the next available time slot after the specified time"""
        # Convert start time to decimal for easier comparison
        start_decimal = after_hour + (after_minute / 60)
        
        # Get booked slots for all interviewers
        booked_by_interviewer = {}
        for interviewer in self.interviewers:
            booked_by_interviewer[interviewer["name"]] = []
        
        for interview in self.scheduled_interviews:
            if interview.interviewer_name in booked_by_interviewer:
                try:
                    # Convert start and end times to hours for easier comparison
                    start_hour, start_min = map(int, interview.start_time.split(':'))
                    end_hour, end_min = map(int, interview.end_time.split(':'))
                    
                    # Mark all hours between start and end as booked (including partial hours)
                    start_decimal = start_hour + (start_min / 60)
                    end_decimal = end_hour + (end_min / 60)
                    
                    # Add a buffer of 15 minutes before and after
                    start_with_buffer = max(0, start_decimal - 0.25)
                    end_with_buffer = min(24, end_decimal + 0.25)
                    
                    booked_by_interviewer[interview.interviewer_name].append((start_with_buffer, end_with_buffer))
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing time in find_next_available_slot: {e}")
                    continue
        
        # Find all possible slots for each interviewer, with enhanced scoring
        scored_slots = []
        
        for interviewer in self.interviewers:
            # Calculate expertise match with this interviewer
            interviewer_expertise = interviewer.get("expertise", [])
            match_score = self.calculate_expertise_match_score(candidate_expertise, interviewer_expertise)
            
            # Skip interviewers with very low match scores
            if match_score < 20:
                continue
                
            # Get interviewer's availability, default to standard business hours if not specified
            availability = interviewer.get("availability", list(range(9, 17)))
                
            # Check each available hour
            for hour in sorted(availability):
                for minute in [0, 15, 30, 45]:
                    slot_decimal = hour + (minute / 60)
                    
                    # Skip slots earlier than requested time
                    if slot_decimal < start_decimal:
                        continue
                        
                    # Skip booked slots
                    booked_times = booked_by_interviewer.get(interviewer["name"], [])
                    slot_end = slot_decimal + 0.75  # 45 minute slot
                    
                    if not any(start <= slot_decimal < end or start < slot_end <= end or 
                              (slot_decimal <= start and slot_end >= end) 
                              for start, end in booked_times):
                        # This slot works - calculate the score
                        # Factors:
                        # 1. Expertise match (0-100) - highest weight
                        # 2. Time closeness (higher if closer to requested time)
                        # 3. Workload balance (higher if interviewer has fewer interviews)
                        
                        # Calculate time difference score (closer = higher score)
                        time_diff = slot_decimal - start_decimal
                        time_score = max(0, 100 - (time_diff * 20))  # Decreases as time difference increases
                        
                        # Calculate workload score
                        current_workload = sum(1 for interview in self.scheduled_interviews 
                                              if interview.interviewer_name == interviewer["name"])
                        workload_score = max(0, 100 - (current_workload * 10))  # Decreases as workload increases
                        
                        # Calculate composite score
                        composite_score = (match_score * 0.6) + (time_score * 0.3) + (workload_score * 0.1)
                        
                        scored_slots.append({
                            "hour": hour,
                            "minute": minute,
                            "interviewer": interviewer,
                            "score": composite_score,
                            "match_score": match_score,
                            "time_score": time_score,
                            "workload_score": workload_score
                        })
        
        # Sort slots by score (highest first)
        scored_slots.sort(key=lambda x: x["score"], reverse=True)
        
        # Return the best slot if available
        if scored_slots:
            best_slot = scored_slots[0]
            logger.info(f"Found best slot with score {best_slot['score']:.1f}: {best_slot['hour']}:{best_slot['minute']} with {best_slot['interviewer']['name']}")
            return best_slot
        
        # If no suitable slot found, create a fallback option
        logger.warning("No suitable time slot found, creating fallback option")
        
        # Find the first interviewer with any availability
        for interviewer in self.interviewers:
            availability = interviewer.get("availability", [])
            if availability:
                # Use the first available hour
                hour = availability[0]
                best_slot = {
                    "hour": hour,
                    "minute": 0,
                    "interviewer": interviewer,
                    "score": 0,
                    "match_score": 0,
                    "time_score": 0,
                    "workload_score": 0
                }
                return best_slot
        
        # If still no slot found, use a default
        if self.interviewers:
            logger.warning("Creating default time slot with first interviewer")
            best_slot = {
                "hour": 9,
                "minute": 0,
                "interviewer": self.interviewers[0],
                "score": 0,
                "match_score": 0,
                "time_score": 0,
                "workload_score": 0
            }
            return best_slot
            
        return None
    
    def _parse_candidate_from_string(self, candidate_str):
        """
        Enhanced candidate parsing with better name extraction and contextual understanding
        
        Args:
            candidate_str: String representation of a candidate resume
            
        Returns:
            Dictionary with parsed candidate information
        """
        # If already a dictionary, just return it with defaults for missing fields
        if isinstance(candidate_str, dict):
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
        
        # Extract ID - try multiple patterns
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
        
        # Extract name - check for specific candidate sections
        candidate_sections = re.findall(r'Candidate\s+\d+:\s+([^(]+)(?:\s*\(Applicant ID (\d+)\))?', candidate_str)
        if candidate_sections:
            for section in candidate_sections:
                candidate_name = section[0].strip()
                if candidate_name and candidate_name.lower() != "unknown":
                    candidate_data["Name"] = candidate_name
                    # If we also found an ID, update it
                    if section[1]:
                        candidate_data["ID"] = section[1]
                    break
        
        # If no name found in candidate sections, try other patterns
        if "Name" not in candidate_data:
            # Look for explicit name fields
            name_patterns = [
                r'Name[:\s]*([^\n]+)',
                r'Candidate\s+Name[:\s]*([^\n]+)',
                r'Applicant\s+Name[:\s]*([^\n]+)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+):'  # Name followed by colon
            ]
            
            for pattern in name_patterns:
                name_match = re.search(pattern, candidate_str, re.IGNORECASE)
                if name_match:
                    candidate_name = name_match.group(1).strip()
                    if candidate_name.lower() != "unknown":
                        candidate_data["Name"] = candidate_name
                        break
        
        # If still no name, check the first few lines for a proper name
        if "Name" not in candidate_data:
            lines = candidate_str.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$', line)
                if name_match and not re.search(r'@|www|\d', line):  # Avoid lines with email/website/numbers
                    candidate_data["Name"] = name_match.group(1)
                    break
        
        # If still no name found, use candidate ID
        if "Name" not in candidate_data or not candidate_data["Name"]:
            candidate_data["Name"] = f"Candidate {candidate_data['ID']}"
        
        # Extract skills
        skills = []
        skill_patterns = [
            r'Skills(?:\s+match)?[:\s]*([^\n.]+)',
            r'Skills:[:\s]*([^\n.]+)',
            r'Technical\s+Skills[:\s]*([^\n.]+)',
            r'candidate_key_skills[:\s]*\[(.*?)\]'
        ]
        
        for pattern in skill_patterns:
            skills_match = re.search(pattern, candidate_str, re.IGNORECASE)
            if skills_match:
                skills_text = skills_match.group(1).strip()
                if skills_text.lower() != "unknown":
                    # Parse skills - handle quoted strings, commas, brackets, etc.
                    if "[" in skills_text or "]" in skills_text:
                        # Try to parse as a list
                        skills_text = skills_text.replace("[", "").replace("]", "")
                        
                    if "," in skills_text:
                        parsed_skills = [s.strip().strip('"\'') for s in skills_text.split(",")]
                        skills.extend([s for s in parsed_skills if s])
                    else:
                        skills.append(skills_text)
        
        # If no skills found yet, try to find common tech keywords
        if not skills:
            tech_keywords = [
                "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "go",
                "react", "angular", "vue", "node", "express", "django", "flask", 
                "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
                "sql", "mysql", "postgresql", "mongodb", "nosql", "elasticsearch",
                "machine learning", "ai", "tensorflow", "pytorch", "nlp", "genai",
                "frontend", "backend", "fullstack", "mobile", "android", "ios",
                "devops", "ci/cd", "jenkins", "git", "github", "agile", "scrum"
            ]
            
            for keyword in tech_keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', candidate_str, re.IGNORECASE):
                    skills.append(keyword)
        
        # Store skills in the candidate data
        if skills:
            candidate_data["Skills"] = skills
        else:
            candidate_data["Skills"] = ["General"]
        
        # Store the original resume text
        candidate_data["Resume"] = candidate_str
        
        return candidate_data
        
    def schedule_interview_with_best_match(self, candidate, params):
        """
        Schedule an interview with the best matching interviewer based on skills
        
        Args:
            candidate: Dictionary with candidate information
            params: Dictionary with scheduling parameters (skills, time, date, etc.)
            
        Returns:
            Dictionary with scheduled interview details or None if scheduling failed
        """
        try:
            # Extract parameters with robust defaults
            candidate_id = str(candidate.get("ID", "")) if candidate else str(random.randint(1000, 9999))
            candidate_name = candidate.get("Name", f"Candidate {candidate_id}") if candidate else f"Candidate {candidate_id}"
            
            # Get candidate skills with proper handling of different formats
            candidate_skills = candidate.get("Skills", []) if candidate else []
            if isinstance(candidate_skills, str):
                candidate_skills = [candidate_skills]
            elif not candidate_skills:
                candidate_skills = ["General"]
            
            requested_time = params.get("interview_time", "9am")
            requested_date = params.get("interview_date", "tomorrow")
            preferred_interviewer = params.get("preferred_interviewer")
            
            # Find the best interviewer or use preferred if specified
            if preferred_interviewer:
                interviewer_obj = next((i for i in self.interviewers if i["name"].lower() == preferred_interviewer.lower()), 
                                    None)
                if not interviewer_obj:
                    # Try matching by partial name
                    interviewer_obj = next((i for i in self.interviewers 
                                          if preferred_interviewer.lower() in i["name"].lower()), 
                                         None)
                                         
                # If still not found, use best match
                if not interviewer_obj:
                    interviewer_obj = self.find_best_interviewer(candidate_skills)
            else:
                # Calculate the best interviewer using enhanced matching
                interviewer_scores = []
                for interviewer in self.interviewers:
                    # Calculate expertise match score
                    expertise_match = self.calculate_expertise_match_score(
                        candidate_skills, 
                        interviewer.get("expertise", [])
                    )
                    
                    # Calculate workload factor (lower workload = higher score)
                    workload = sum(1 for interview in self.scheduled_interviews 
                                 if interview.interviewer_name == interviewer["name"])
                    workload_factor = max(0, 1.0 - (workload * 0.1))
                    
                    # Calculate composite score
                    total_score = expertise_match * workload_factor
                    
                    interviewer_scores.append((interviewer, total_score, expertise_match))
                
                # Sort by total score (highest first)
                interviewer_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Use top interviewer if scores exist
                if interviewer_scores:
                    interviewer_obj = interviewer_scores[0][0]
                    expertise_score = interviewer_scores[0][2]
                    logger.info(f"Selected best interviewer {interviewer_obj['name']} with expertise match {expertise_score:.1f}%")
                else:
                    interviewer_obj = self.interviewers[0] if self.interviewers else None
            
            # Schedule the interview - validate params first
            if not requested_time:
                requested_time = "9am"  # Default time
                
            # Handle various date formats
            if not requested_date:
                requested_date = "tomorrow"  # Default date
            
            # Schedule the interview
            try:
                interview_slot = self.schedule_interview(
                    candidate_id, 
                    candidate_name, 
                    candidate_skills,
                    requested_time,
                    requested_date,
                    interviewer_obj.get("name") if interviewer_obj else None
                )
                
                if not interview_slot:
                    logger.warning("Failed to schedule interview, creating manual fallback")
                    # Create a minimal fallback slot
                    today = datetime.datetime.now()
                    interview_date = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                    
                    # Fallback to first interviewer if needed
                    if not interviewer_obj and self.interviewers:
                        interviewer_obj = self.interviewers[0]
                    
                    if interviewer_obj:
                        # Create a basic interview slot
                        interview_slot = InterviewSlot(
                            candidate_id=candidate_id,
                            candidate_name=candidate_name,
                            interviewer_name=interviewer_obj["name"],
                            date=interview_date,
                            start_time="09:00",
                            end_time="09:45",
                            meeting_link=self.generate_meeting_link(),
                            email=interviewer_obj.get("email", "interviewer@company.com"),
                            status="scheduled"
                        )
                        
                        self.scheduled_interviews.append(interview_slot)
                
                # Convert InterviewSlot to dictionary for easier handling
                return {
                    "candidate_id": interview_slot.candidate_id,
                    "candidate_name": interview_slot.candidate_name,
                    "candidate_key_skills": candidate_skills,
                    "interviewer_id": next((i["id"] for i in self.interviewers if i["name"] == interview_slot.interviewer_name), ""),
                    "interviewer_name": interview_slot.interviewer_name,
                    "date": interview_slot.date,
                    "start_time": interview_slot.start_time,
                    "end_time": interview_slot.end_time,
                    "meeting_link": interview_slot.meeting_link,
                    "email": interview_slot.email,
                    "status": interview_slot.status
                }
            except Exception as e:
                logger.error(f"Error in schedule_interview: {e}")
                # Create fallback interview
                today = datetime.datetime.now()
                interview_date = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                start_time = "09:00"
                end_time = "09:45"
                
                # Use the first interviewer as fallback
                fallback_interviewer = interviewer_obj or (self.interviewers[0] if self.interviewers else None)
                
                if fallback_interviewer:
                    # Create a basic interview as fallback
                    meeting_link = self.generate_meeting_link()
                    
                    return {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "candidate_key_skills": candidate_skills,
                        "interviewer_id": fallback_interviewer.get("id", "default"),
                        "interviewer_name": fallback_interviewer.get("name", "Default Interviewer"),
                        "date": interview_date,
                        "start_time": start_time,
                        "end_time": end_time,
                        "meeting_link": meeting_link,
                        "email": fallback_interviewer.get("email", "interviewer@company.com"),
                        "status": "scheduled"
                    }
                else:
                    return None
        except Exception as e:
            logger.error(f"Error scheduling interview: {e}")
            # Return a minimal response that won't cause the interface to crash
            return {
                "candidate_id": "error",
                "candidate_name": "Error Scheduling",
                "interviewer_name": "Default Interviewer",
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "start_time": "09:00",
                "end_time": "09:45",
                "meeting_link": "https://meet.google.com/error",
                "email": "error@company.com",
                "status": "error"
            }
    # Add this method to the InterviewScheduler class in interview_agent.py

    def get_available_interviewers(self):
        """
        Get the list of available interviewers with their expertise and availability
        
        Returns:
            List of interviewer dictionaries
        """
        # Return the existing interviewers list
        return self.interviewers