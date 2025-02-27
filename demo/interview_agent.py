# interview_agent.py
import os
import json
import random
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class InterviewSlot:
    candidate_id: str
    candidate_name: str
    interviewer_name: str
    date: str
    start_time: str
    end_time: str
    meeting_link: str
    email: str
    status: str = "pending"  # pending, scheduled, sent

class InterviewScheduler:
    def __init__(self, interviewers=None):
        # Default interviewers if none provided
        self.interviewers = interviewers or [
            {"name": "Alice Chen", "email": "alice.chen@company.com", "expertise": ["GenAI", "NLP"], 
             "availability": [9, 10, 11, 14, 15, 16]},
            {"name": "Bob Smith", "email": "bob.smith@company.com", "expertise": ["GenAI", "MLOps"], 
             "availability": [10, 11, 12, 13, 14]},
            {"name": "Carol Wong", "email": "carol.wong@company.com", "expertise": ["Fullstack", "Frontend"], 
             "availability": [9, 10, 13, 14, 15]},
            {"name": "David Maguire", "email": "david.maguire@company.com", "expertise": ["Fullstack", "Backend"], 
             "availability": [11, 12, 13, 16, 17]},
            {"name": "Eva Martinez", "email": "eva.martinez@company.com", "expertise": ["GenAI", "Data Science"], 
             "availability": [9, 10, 11, 14, 15]}
        ]
        self.scheduled_interviews = []
        
    def generate_meeting_link(self):
        """Generate a mock Google Meet link"""
        meeting_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3)) + '-' + \
                     ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4)) + '-' + \
                     ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))
        return f"https://meet.google.com/{meeting_id}"
    
    def find_best_interviewer(self, candidate_expertise):
        """Find the best interviewer based on expertise match"""
        best_match = None
        best_score = -1
        
        for interviewer in self.interviewers:
            score = sum(1 for skill in candidate_expertise if skill in interviewer["expertise"])
            if score > best_score:
                best_score = score
                best_match = interviewer
                
        # If no match found, return first interviewer as fallback
        return best_match or self.interviewers[0]
    
    def parse_time_request(self, time_str):
        """Parse time string like '9am' or '2:30pm'"""
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
        else:
            # Assume 24-hour format
            if ":" in time_str:
                hour, minute = map(int, time_str.split(":"))
            else:
                hour = int(time_str)
                minute = 0
        
        return hour, minute
    
    def schedule_interview(self, candidate_id, candidate_name, candidate_expertise, 
                      requested_time=None, requested_date=None, interviewer=None):
        """Schedule an interview with the specified parameters"""
        # Generate today's date if not specified
        if not requested_date or requested_date.lower() in ["tomorrow", "today"]:
            today = datetime.datetime.now()
            if requested_date and requested_date.lower() == "today":
                days_ahead = 0
            else:  # Tomorrow is default
                days_ahead = 1 if today.weekday() < 4 else 3  # Skip to Monday if today is Friday
            requested_date = (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif requested_date.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            # Handle day names
            today = datetime.datetime.now()
            day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
                    "friday": 4, "saturday": 5, "sunday": 6}
            target_day = day_map[requested_date.lower()]
            days_ahead = (target_day - today.weekday()) % 7
            if days_ahead == 0:  # If today is the requested day, schedule for next week
                days_ahead = 7
            requested_date = (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Find best interviewer if not specified
        if not interviewer:
            interviewer_obj = self.find_best_interviewer(candidate_expertise)
        else:
            interviewer_obj = next((i for i in self.interviewers if i["name"].lower() == interviewer.lower()), 
                                self.find_best_interviewer(candidate_expertise))
        
        # Get already scheduled times for this interviewer on this date
        booked_times = []
        for interview in self.scheduled_interviews:
            if interview.interviewer_name == interviewer_obj["name"] and interview.date == requested_date:
                # Convert start and end times to hours for easier comparison
                start_hour, start_min = map(int, interview.start_time.split(':'))
                end_hour, end_min = map(int, interview.end_time.split(':'))
                
                # Mark all hours between start and end as booked (including partial hours)
                start_decimal = start_hour + (start_min / 60)
                end_decimal = end_hour + (end_min / 60)
                
                # Add a buffer of 15 minutes before and after
                start_with_buffer = max(0, start_decimal - 0.25)
                end_with_buffer = min(24, end_decimal + 0.25)
                
                booked_times.append((start_with_buffer, end_with_buffer))
        
        # Parse requested time or find first available slot
        if requested_time:
            hour, minute = self.parse_time_request(requested_time)
            requested_decimal = hour + (minute / 60)
            
            # Check if the time is already booked
            is_booked = any(start <= requested_decimal < end for start, end in booked_times)
            
            # Check if interviewer is available at this time and time isn't booked
            if hour not in interviewer_obj["availability"] or is_booked:
                # Find closest available time that isn't booked
                available_hours = sorted(interviewer_obj["availability"])
                available_slots = []
                
                for h in available_hours:
                    # Check if any 45-minute slot starting at this hour is free
                    for m in [0, 15, 30]:
                        slot_start = h + (m / 60)
                        slot_end = slot_start + 0.75  # 45 minutes
                        
                        if not any(start <= slot_start < end or start < slot_end <= end or 
                                (slot_start <= start and slot_end >= end) for start, end in booked_times):
                            available_slots.append((h, m))
                
                if available_slots:
                    # Choose the slot closest to requested time
                    requested_decimal = hour + (minute / 60)
                    closest_slot = min(available_slots, 
                                    key=lambda x: abs((x[0] + (x[1]/60)) - requested_decimal))
                    hour, minute = closest_slot
                else:
                    # No available slots with current interviewer, try another
                    for other_interviewer in self.interviewers:
                        if other_interviewer != interviewer_obj:
                            # Check if this interviewer has expertise match
                            score = sum(1 for skill in candidate_expertise if skill in other_interviewer["expertise"])
                            if score > 0:
                                # Use this interviewer instead
                                interviewer_obj = other_interviewer
                                hour = interviewer_obj["availability"][0]
                                minute = 0
                                break
        else:
            # Use first available time slot that isn't booked
            available_found = False
            for h in sorted(interviewer_obj["availability"]):
                for m in [0, 15, 30]:
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
                # Try the next day
                tomorrow = datetime.datetime.strptime(requested_date, "%Y-%m-%d") + datetime.timedelta(days=1)
                requested_date = tomorrow.strftime("%Y-%m-%d")
                hour = interviewer_obj["availability"][0]
                minute = 0
        
        # Format times for display
        start_time = f"{hour:02d}:{minute:02d}"
        end_hour = hour + ((minute + 45) // 60)
        end_minute = (minute + 45) % 60
        end_time = f"{end_hour:02d}:{end_minute:02d}"
        
        # Create interview slot
        meeting_link = self.generate_meeting_link()
        interview = InterviewSlot(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            interviewer_name=interviewer_obj["name"],
            date=requested_date,
            start_time=start_time,
            end_time=end_time,
            meeting_link=meeting_link,
            email=interviewer_obj["email"],
            status="scheduled"
        )
        
        self.scheduled_interviews.append(interview)
        return interview
    
    def get_scheduled_interviews(self):
        """Return all scheduled interviews"""
        return self.scheduled_interviews
    
    def get_email_draft(self, interview):
        """Generate an email draft for the interview"""
        email_draft = f"""
Subject: Interview Scheduled: {interview.candidate_name} - {interview.date} at {interview.start_time}

Dear {interview.candidate_name},

We are pleased to invite you to an interview for the position at our company.

Interview Details:
- Date: {interview.date}
- Time: {interview.start_time} - {interview.end_time}
- Format: Video Conference
- Meeting Link: {interview.meeting_link}
- Interviewer: {interview.interviewer_name}

Please prepare to discuss your experience and be ready for a short technical assessment.
If you need to reschedule, please let us know at least 24 hours in advance.

We look forward to speaking with you!

Best regards,
HR Team
Company Name
        """
        return email_draft
    
    def send_email(self, interview_id):
        """Mock sending an email for the given interview"""
        for interview in self.scheduled_interviews:
            if interview.candidate_id == interview_id:
                # In a real implementation, this would connect to an email service
                # For now we'll just update the status
                interview.status = "sent"
                return True
        return False

    # Updates to InterviewScheduler class

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
        
        # Validate input lists have the same length
        if len(candidates) != len(requested_times):
            raise ValueError("Number of candidates must match number of requested times")
        
        # Sort by requested time to handle earlier times first
        scheduling_requests = list(zip(candidates, requested_times))
        
        # Process each candidate
        for candidate, requested_time in scheduling_requests:
            candidate_id = candidate.get("ID")
            candidate_name = candidate.get("Name", f"Candidate {candidate_id}")
            candidate_expertise = candidate.get("Skills", ["Fullstack"])
            
            # Parse the requested time
            try:
                hour, minute = self.parse_time_request(requested_time)
                time_key = f"{hour:02d}:{minute:02d}"
            except:
                # Use default time if parsing fails
                hour, minute = 9, 0
                time_key = "09:00"
            
            # Check if we already have an interviewer allocated at this time
            if time_key in interviewer_allocations:
                # Find a different interviewer with matching skills
                used_interviewer = interviewer_allocations[time_key]
                available_interviewers = [
                    i for i in self.interviewers 
                    if i["name"] != used_interviewer["name"] and 
                    hour in i["availability"] and
                    any(skill in i["expertise"] for skill in candidate_expertise)
                ]
                
                if available_interviewers:
                    # Use the best matching available interviewer
                    interviewer = max(
                        available_interviewers,
                        key=lambda i: sum(1 for skill in candidate_expertise if skill in i["expertise"])
                    )
                else:
                    # No other interviewer available at this time, find next available slot
                    next_slot = self.find_next_available_slot(hour, minute, candidate_expertise)
                    if next_slot:
                        hour, minute = next_slot["hour"], next_slot["minute"]
                        time_key = f"{hour:02d}:{minute:02d}"
                        interviewer = next_slot["interviewer"]
                    else:
                        # Fallback: move to next day
                        hour = self.interviewers[0]["availability"][0]
                        minute = 0
                        interviewer = self.find_best_interviewer(candidate_expertise)
            else:
                # Find best interviewer with availability at this time
                available_interviewers = [
                    i for i in self.interviewers 
                    if hour in i["availability"] and
                    any(skill in i["expertise"] for skill in candidate_expertise)
                ]
                
                if available_interviewers:
                    interviewer = max(
                        available_interviewers,
                        key=lambda i: sum(1 for skill in candidate_expertise if skill in i["expertise"])
                    )
                else:
                    # No interviewer available, find closest available time
                    next_slot = self.find_next_available_slot(hour, minute, candidate_expertise)
                    if next_slot:
                        hour, minute = next_slot["hour"], next_slot["minute"]
                        time_key = f"{hour:02d}:{minute:02d}"
                        interviewer = next_slot["interviewer"]
                    else:
                        # Fallback to first available interviewer
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
            interview = InterviewSlot(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                interviewer_name=interviewer["name"],
                date=requested_date,
                start_time=start_time,
                end_time=end_time,
                meeting_link=meeting_link,
                email=interviewer["email"],
                status="scheduled"
            )
            
            self.scheduled_interviews.append(interview)
            scheduled.append(interview)
        
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
        
        # Find all possible slots for each interviewer
        best_slot = None
        best_slot_time = float('inf')  # Earliest time is best
        
        for interviewer in self.interviewers:
            # Skip interviewers without matching expertise
            if not any(skill in interviewer["expertise"] for skill in candidate_expertise):
                continue
                
            # Check each available hour
            for hour in sorted(interviewer["availability"]):
                for minute in [0, 15, 30, 45]:
                    slot_decimal = hour + (minute / 60)
                    
                    # Skip slots earlier than requested time
                    if slot_decimal < start_decimal:
                        continue
                        
                    # Skip booked slots
                    booked_times = booked_by_interviewer[interviewer["name"]]
                    slot_end = slot_decimal + 0.75  # 45 minute slot
                    
                    if not any(start <= slot_decimal < end or start < slot_end <= end or 
                            (slot_decimal <= start and slot_end >= end) 
                            for start, end in booked_times):
                        # This slot works and is available
                        if slot_decimal < best_slot_time:
                            best_slot_time = slot_decimal
                            best_slot = {
                                "hour": hour,
                                "minute": minute,
                                "interviewer": interviewer
                            }
        
        return best_slot

    def generate_bulk_invitation_drafts(self, scheduled_interviews):
        """Generate separate email drafts for multiple scheduled interviews"""
        invitation_drafts = []
        
        for interview in scheduled_interviews:
            email_draft = self.get_email_draft(interview)
            invitation_drafts.append({
                "candidate_id": interview.candidate_id,
                "candidate_name": interview.candidate_name,
                "email_draft": email_draft
            })
        
        return invitation_drafts