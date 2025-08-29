# Prompt Engineering & Pydantic for LLM Data Validation

A comprehensive FastAPI project demonstrating prompt engineering techniques and structured LLM output validation using Pydantic.

## 🎯 Learning Objectives (1 Hour Class)

This project covers:
- **Prompt Engineering Fundamentals**
- **Prompt Types**: Zero-shot, Few-shot, Chain-of-thought
- **Pydantic for Structuring Outputs**
- **Output Parsers for LLM Responses**
- **Practical Implementation**: Task extraction API

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure OpenAI API Key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Run the Application

```bash
python main.py
```

The API will be available at: `http://localhost:8000`

### 3. Explore the API

- **API Documentation**: `http://localhost:8000/docs`
- **Interactive Testing**: Use the Swagger UI

## 📚 Core Concepts Covered

### 1. What is Prompt Engineering?

Prompt engineering is the practice of designing and optimizing prompts to get better responses from Large Language Models (LLMs). It involves:

- **Clarity**: Making instructions clear and unambiguous
- **Context**: Providing relevant background information
- **Structure**: Organizing prompts for optimal understanding
- **Examples**: Using demonstrations to guide model behavior

### 2. Prompt Types Implemented

#### Zero-Shot Prompting
```python
# Direct instruction without examples
"Extract task information from the following text and return it as JSON..."
```

#### Few-Shot Prompting
```python
# Includes examples to guide the model
"Here are examples of task extraction:
Example 1: ...
Example 2: ...
Now extract from: ..."
```

#### Chain-of-Thought Prompting
```python
# Encourages step-by-step reasoning
"Let's extract task information step by step:
1. First, identify the main action...
2. Look for time indicators..."
```

### 3. Pydantic for Structure

Pydantic models ensure type safety and data validation:

```python
class Task(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    due_date: Optional[date] = Field(None)
    priority: Priority = Field(Priority.MEDIUM)
    description: Optional[str] = Field(None, max_length=500)
    
    @validator('due_date')
    def validate_due_date(cls, v):
        if v and v < date.today():
            raise ValueError('Due date cannot be in the past')
        return v
```

### 4. Output Parsers

Custom parsers extract and validate LLM responses:

```python
class TaskOutputParser:
    @staticmethod
    def parse_json_response(response: str) -> dict:
        # Extract JSON from LLM response using regex
        
    @staticmethod
    def validate_and_create_task(data: dict) -> Task:
        # Validate and create Pydantic model
```

## 🛠 API Endpoints

### 1. Extract Task (Main Demo)
```http
POST /extract-task
```

**Request Body:**
```json
{
  "text": "Complete the quarterly report by Friday, it's very important",
  "prompt_type": "few-shot"
}
```

**Response:**
```json
{
  "title": "Complete quarterly report",
  "due_date": "2024-01-19",
  "priority": "high",
  "description": "Quarterly report completion"
}
```

### 2. Prompt Examples (Educational)
```http
POST /prompt-examples
```

Shows the actual prompts sent to the LLM without making API calls.

## 🎓 Class Exercise Flow (1 Hour)

### Phase 1: Understanding (15 minutes)
1. **Explore the API documentation** at `/docs`
2. **Review the code structure** in `main.py`
3. **Understand Pydantic models** and validation

### Phase 2: Prompt Engineering (20 minutes)
1. **Test different prompt types** using `/prompt-examples`
2. **Compare zero-shot vs few-shot vs chain-of-thought**
3. **Analyze prompt effectiveness**

### Phase 3: Hands-on Practice (20 minutes)
1. **Configure your OpenAI API key**
2. **Test the `/extract-task` endpoint** with various inputs:
   - "Schedule a meeting with the team next Tuesday"
   - "Buy groceries and clean the house this weekend"
   - "Submit the project proposal by end of month - URGENT"
3. **Observe structured outputs** and validation

### Phase 4: Customization (5 minutes)
1. **Modify the Task model** to add new fields
2. **Create custom validation rules**
3. **Test with your changes**

## 🧪 Example Test Cases

```python
# Test different text inputs
test_cases = [
    "Complete the quarterly report by Friday, it's very important",
    "Buy groceries sometime this week",
    "Schedule a meeting with the team next Tuesday at 2 PM",
    "Submit the project proposal by end of month - URGENT",
    "Call mom"
]

# Test different prompt types
prompt_types = ["zero-shot", "few-shot", "chain-of-thought"]
```

## 🔧 Key Features Demonstrated

- ✅ **Type Safety** with Pydantic models
- ✅ **Data Validation** with custom validators
- ✅ **Enum Usage** for controlled vocabularies
- ✅ **Error Handling** for invalid inputs
- ✅ **API Documentation** with FastAPI
- ✅ **Environment Configuration** with python-dotenv
- ✅ **Structured Logging** and responses

## 🎯 Learning Outcomes

After this class, you will understand:

1. **How to design effective prompts** for different scenarios
2. **When to use different prompt types** (zero-shot vs few-shot vs CoT)
3. **How to structure LLM outputs** using Pydantic
4. **How to validate and parse** LLM responses reliably
5. **How to build production-ready APIs** with structured data

## 🚀 Next Steps

- Experiment with different OpenAI models (GPT-4, etc.)
- Add more complex validation rules
- Implement batch processing for multiple tasks
- Add authentication and rate limiting
- Deploy to production with proper error handling

## 📝 Notes

- This project uses OpenAI's legacy API format for compatibility
- Ensure your OpenAI API key has sufficient credits
- The examples are designed for educational purposes
- Production use would require additional error handling and security measures