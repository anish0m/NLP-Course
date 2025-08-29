from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime, date, timedelta
from enum import Enum
import openai
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Prompt Engineering & Pydantic LLM Validation",
    description="Learn prompt engineering and structured LLM outputs with Pydantic",
    version="1.0.0"
)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic Models for structured outputs
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Task(BaseModel):
    """Structured task model with validation"""
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    due_date: Optional[date] = Field(None, description="Task due date")
    priority: Priority = Field(Priority.MEDIUM, description="Task priority level")
    description: Optional[str] = Field(None, max_length=500, description="Task description")
    
    @validator('due_date')
    def validate_due_date(cls, v):
        # Auto-adjust past dates to today's date
        if v and v < date.today():
            return date.today()
        return v

class TaskList(BaseModel):
    """Container for multiple tasks"""
    tasks: List[Task]
    total_count: int

class PromptRequest(BaseModel):
    """Request model for prompt engineering examples"""
    text: str = Field(..., description="Input text to process")
    prompt_type: Literal["zero-shot", "few-shot", "chain-of-thought"] = Field(
        "zero-shot", description="Type of prompt to use"
    )

class PromptResponse(BaseModel):
    """Response model for prompt results"""
    prompt_used: str
    raw_response: str
    structured_output: Optional[dict] = None

# Prompt Engineering Templates
class PromptTemplates:
    
    @staticmethod
    def zero_shot_task_extraction(text: str) -> str:
        """Zero-shot prompt for task extraction"""
        return f"""
Extract task information from the following text and return it as JSON with these fields:
- title: string (required)
- due_date: string in YYYY-MM-DD format (optional)
- priority: one of "low", "medium", "high", "urgent" (default: "medium")
- description: string (optional)

Text: "{text}"

JSON:
"""
    
    @staticmethod
    def few_shot_task_extraction(text: str) -> str:
        """Few-shot prompt with examples"""
        return f"""
Extract task information from text and return as JSON. Here are examples:

Example 1:
Text: "Complete the quarterly report by Friday, it's very important"
JSON: {{"title": "Complete quarterly report", "due_date": "2024-01-19", "priority": "high", "description": "Quarterly report completion"}}

Example 2:
Text: "Buy groceries sometime this week"
JSON: {{"title": "Buy groceries", "due_date": null, "priority": "low", "description": "Weekly grocery shopping"}}

Now extract from:
Text: "{text}"
JSON:
"""
    
    @staticmethod
    def chain_of_thought_task_extraction(text: str) -> str:
        """Chain-of-thought prompt with reasoning"""
        return f"""
Let's extract task information step by step:

1. First, identify the main action or task in the text
2. Look for any time indicators or deadlines - be very specific about dates:
   - For "end of month" → use the last day of the current month
   - For "next week" → calculate the exact date 7 days from today
   - For "tomorrow" → calculate the exact date for tomorrow
   - For "Friday" → find the date of the next Friday
   - Always convert to YYYY-MM-DD format with actual dates (not the literal string "YYYY-MM-DD")
3. Assess the urgency or priority based on language used (must be one of: low, medium, high, urgent)
4. Extract any additional context or description

Text: "{text}"

Step-by-step analysis:
1. Main task: [identify the core action]
2. Timeline analysis:
   - Mentioned deadline: [extract any deadline mentioned]
   - Today's date: [today's date in YYYY-MM-DD]
   - Calculated due date: [convert the deadline to an actual date in YYYY-MM-DD format]
3. Priority assessment: [analyze urgency indicators]
4. Additional context: [extract relevant details]

Based on this analysis, create a structured JSON with these exact field names:
- title: The main task title
- due_date: The deadline in YYYY-MM-DD format with actual date values (or null if not specified)
- priority: Must be exactly one of: "low", "medium", "high", "urgent" (lowercase only)
- description: Additional context or details

JSON output:
{{
  "title": "...",
  "due_date": "2023-12-31",
  "priority": "...",
  "description": "..."
}}
"""

# Output Parser for LLM responses
class TaskOutputParser:
    """Parser to extract structured data from LLM responses"""
    
    @staticmethod
    def parse_json_response(response: str) -> dict:
        """Extract JSON from LLM response"""
        # Try to find JSON in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response)
        
        if matches:
            try:
                return json.loads(matches[-1])  # Take the last JSON found
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse the entire response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            raise ValueError("Could not extract valid JSON from response")
    
    @staticmethod
    def validate_and_create_task(data: dict) -> Task:
        """Validate parsed data and create Task object"""
        # Normalize field names and handle common LLM output variations
        normalized_data = {}
        
        # Handle different field name variations
        for key, value in data.items():
            # Map common variations to expected field names
            if key in ['title', 'task', 'task_title', 'main_task']:
                normalized_data['title'] = value
            elif key in ['due_date', 'deadline', 'due', 'date']:
                # Handle special date cases
                if isinstance(value, str):
                    if "end of month" in value.lower():
                        # Calculate end of current month
                        today = date.today()
                        # Get the last day of the current month
                        if today.month == 12:
                            last_day = date(today.year + 1, 1, 1) - timedelta(days=1)
                        else:
                            # Get first day of next month, then subtract 1 day
                            next_month_first = date(today.year, today.month + 1, 1)
                            last_day = next_month_first - timedelta(days=1)
                        normalized_data['due_date'] = last_day
                    else:
                        normalized_data['due_date'] = value
                else:
                    normalized_data['due_date'] = value
            elif key in ['priority', 'importance', 'urgency']:
                # Normalize priority values to lowercase
                if isinstance(value, str):
                    if value.lower() in ['low', 'medium', 'high', 'urgent']:
                        normalized_data['priority'] = value.lower()
                    elif value.upper() == 'URGENT':
                        normalized_data['priority'] = 'urgent'
                else:
                    normalized_data['priority'] = value
            elif key in ['description', 'details', 'notes', 'context']:
                normalized_data['description'] = value
        
        # Ensure required fields are present
        if 'title' not in normalized_data and 'main_task' in data:
            normalized_data['title'] = data['main_task']
            
        try:
            return Task(**normalized_data)
        except Exception as e:
            raise ValueError(f"Invalid task data: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    """Welcome endpoint with API information"""
    return {
        "message": "Prompt Engineering & Pydantic LLM Validation API",
        "topics_covered": [
            "Zero-shot prompting",
            "Few-shot prompting", 
            "Chain-of-thought prompting",
            "Pydantic data validation",
            "LLM output parsing"
        ],
        "endpoints": [
            "/docs - API documentation",
            "/extract-task - Extract structured task from text",
            "/prompt-examples - Try different prompt types"
        ]
    }

@app.post("/extract-task", response_model=Task)
async def extract_task(request: PromptRequest):
    """Extract structured task information from text using different prompt types"""
    
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    # Print debug information
    print(f"Processing request text: {request.text}")
    
    # Initialize variables for date calculation
    text_lower = request.text.lower()
    today = date.today()
    calculated_special_date = None
    
    print(f"Text contains end of month pattern. Original text: '{request.text}', Calculated: {calculated_special_date}")

    # Handle end of month with expanded pattern matching
    if any(phrase in text_lower for phrase in ["end of month", "by end of month", "end of the month", "by the end of month", "month end", "month-end", "monthend", "eom", "end-of-month"]):
        # Calculate end of current month
        # Get the last day of the current month
        if today.month == 12:
            calculated_special_date = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            calculated_special_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
        print(f"Detected 'end of month', calculated date: {calculated_special_date}")
    
    # Handle this week
    elif "this week" in text_lower or "end of this week" in text_lower or "end of week" in text_lower:
        # Calculate the end of the current week (Sunday)
        today_weekday = today.weekday()  # 0 is Monday, 6 is Sunday
        days_until_sunday = 6 - today_weekday
        calculated_special_date = today + timedelta(days=days_until_sunday)
        print(f"Detected 'this week', calculated date (Sunday): {calculated_special_date}")
    
    # Handle all weekday patterns in a more general way
    weekday_patterns = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2, "weds": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6
    }
    
    # Check for any weekday pattern
    for day_name, day_num in weekday_patterns.items():
        if day_name in text_lower or f"on {day_name}" in text_lower or f"by {day_name}" in text_lower:
            # Calculate the date for the next occurrence of this day
            today_weekday = today.weekday()
            
            # Calculate days until the next occurrence
            days_until = (day_num - today_weekday) % 7
            
            # If today is the day and we want the next occurrence, add 7 days
            if days_until == 0:
                days_until = 7
                
            calculated_special_date = today + timedelta(days=days_until)
            print(f"Detected '{day_name}', calculated date: {calculated_special_date}")
            break
    
    # Select prompt based on type
    if request.prompt_type == "zero-shot":
        prompt = PromptTemplates.zero_shot_task_extraction(request.text)
    elif request.prompt_type == "few-shot":
        prompt = PromptTemplates.few_shot_task_extraction(request.text)
    else:  # chain-of-thought
        prompt = PromptTemplates.chain_of_thought_task_extraction(request.text)
    
    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts task information from text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response content
        raw_response = response.choices[0].message.content
        
        # Parse the response
        parsed_data = TaskOutputParser.parse_json_response(raw_response)
        
        # FORCE OVERRIDE the due_date with our calculated special date if we have one
        if calculated_special_date:
            parsed_data["due_date"] = calculated_special_date.isoformat()
            print(f"FORCED special date override: {calculated_special_date.isoformat()}")
            print(f"Final parsed_data: {parsed_data}")
        
        # Validate and create Task object
        task = TaskOutputParser.validate_and_create_task(parsed_data)
        
        return task
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    

@app.post("/prompt-examples", response_model=PromptResponse)
async def demonstrate_prompts(request: PromptRequest):
    """Demonstrate different prompt types without calling OpenAI"""
    
    # Select prompt based on type
    if request.prompt_type == "zero-shot":
        prompt = PromptTemplates.zero_shot_task_extraction(request.text)
    elif request.prompt_type == "few-shot":
        prompt = PromptTemplates.few_shot_task_extraction(request.text)
    else:  # chain-of-thought
        prompt = PromptTemplates.chain_of_thought_task_extraction(request.text)
    
    return PromptResponse(
        prompt_used=prompt,
        raw_response="This endpoint shows the prompt that would be sent to the LLM",
        structured_output=None
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)