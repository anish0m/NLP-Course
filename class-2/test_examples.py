#!/usr/bin/env python3
"""
Test Examples for Prompt Engineering & Pydantic LLM Validation

This script demonstrates how to test the API endpoints and understand
the different prompt engineering techniques.
"""

import requests
import json
from typing import List, Dict

# API Base URL
BASE_URL = "http://localhost:8002"

# Test cases for different scenarios
TEST_CASES = [
    {
        "text": "Complete the quarterly report by Friday, it's very important",
        "expected_priority": "high",
        "description": "Urgent business task with deadline"
    },
    {
        "text": "Buy groceries sometime this week",
        "expected_priority": "low",
        "description": "Casual personal task"
    },
    {
        "text": "Schedule a meeting with the team next Tuesday at 2 PM",
        "expected_priority": "medium",
        "description": "Work task with specific time"
    },
    {
        "text": "Submit the project proposal by end of month - URGENT",
        "expected_priority": "urgent",
        "description": "Critical deadline with explicit urgency"
    },
    {
        "text": "Call mom",
        "expected_priority": "medium",
        "description": "Simple task without context"
    }
]

PROMPT_TYPES = ["zero-shot", "few-shot", "chain-of-thought"]

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running successfully")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

def demonstrate_prompt_types(text: str):
    """Demonstrate different prompt types for the same input"""
    print(f"\n🔍 Analyzing text: '{text}'")
    print("=" * 60)
    
    for prompt_type in PROMPT_TYPES:
        print(f"\n📝 {prompt_type.upper()} PROMPT:")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{BASE_URL}/prompt-examples",
                json={"text": text, "prompt_type": prompt_type}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(data["prompt_used"])
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {str(e)}")

def test_task_extraction(text: str, prompt_type: str = "few-shot"):
    """Test actual task extraction with OpenAI API"""
    try:
        response = requests.post(
            f"{BASE_URL}/extract-task",
            json={"text": text, "prompt_type": prompt_type}
        )
        
        if response.status_code == 200:
            task = response.json()
            print(f"✅ Extracted Task:")
            print(f"   Title: {task['title']}")
            print(f"   Due Date: {task.get('due_date', 'Not specified')}")
            print(f"   Priority: {task['priority']}")
            print(f"   Description: {task.get('description', 'Not specified')}")
            return task
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def run_comprehensive_test():
    """Run comprehensive tests on all test cases"""
    print("🧪 COMPREHENSIVE TESTING")
    print("=" * 50)
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n📋 Test Case {i}: {test_case['description']}")
        print(f"Input: '{test_case['text']}'")
        
        # Test with few-shot prompting (usually most reliable)
        task = test_task_extraction(test_case['text'], "few-shot")
        
        if task:
            # Check if priority matches expectation
            expected = test_case['expected_priority']
            actual = task['priority']
            
            if actual == expected:
                print(f"✅ Priority prediction correct: {actual}")
            else:
                print(f"⚠️  Priority mismatch - Expected: {expected}, Got: {actual}")
            
            results.append({
                "test_case": i,
                "success": True,
                "expected_priority": expected,
                "actual_priority": actual,
                "task": task
            })
        else:
            results.append({
                "test_case": i,
                "success": False,
                "expected_priority": test_case['expected_priority'],
                "actual_priority": None,
                "task": None
            })
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n📊 SUMMARY: {successful}/{len(results)} tests successful")
    
    return results

def interactive_demo():
    """Interactive demonstration for class use"""
    print("🎓 INTERACTIVE DEMO")
    print("=" * 30)
    print("Enter text to extract tasks from (or 'quit' to exit)")
    
    while True:
        text = input("\n📝 Enter task text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
            
        if not text:
            continue
            
        print("\nChoose prompt type:")
        print("1. Zero-shot")
        print("2. Few-shot")
        print("3. Chain-of-thought")
        
        choice = input("Enter choice (1-3, default=2): ").strip()
        
        prompt_type_map = {
            '1': 'zero-shot',
            '2': 'few-shot', 
            '3': 'chain-of-thought'
        }
        
        prompt_type = prompt_type_map.get(choice, 'few-shot')
        
        print(f"\n🔄 Using {prompt_type} prompting...")
        test_task_extraction(text, prompt_type)

def main():
    """Main function to run demonstrations"""
    print("🚀 PROMPT ENGINEERING & PYDANTIC DEMO")
    print("=" * 50)
    
    # Check API health
    if not test_api_health():
        print("\n💡 To start the API, run: python main.py")
        return
    
    while True:
        print("\n🎯 Choose a demo:")
        print("1. Show prompt examples (no API calls)")
        print("2. Test task extraction (requires OpenAI API key)")
        print("3. Run comprehensive tests")
        print("4. Interactive demo")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            sample_text = "Complete the quarterly report by Friday, it's very important"
            demonstrate_prompt_types(sample_text)
            
        elif choice == '2':
            sample_text = "Schedule a meeting with the team next Tuesday at 2 PM"
            print(f"\n🧪 Testing with: '{sample_text}'")
            test_task_extraction(sample_text)
            
        elif choice == '3':
            run_comprehensive_test()
            
        elif choice == '4':
            interactive_demo()
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()