from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
import re


class EmailState(TypedDict):
    email_content: str
    sender: str
    priority: str
    sentiment: str
    category: str
    entities: List[str]
    route: str
    reply: str


def classify_email(state: EmailState) -> EmailState:
    "Classify Email priority, sentiment, and category."

    content = state["email_content"].lower()

    # Priority classification (very naive)
    if any(word in content for word in ["urgent", "asap", "emergency"]):
        priority = "high"
    elif any(word in content for word in ["soon", "important"]):
        priority = "medium"
    else:
        priority = "low"

    # Sentiment analysis (very naive)
    if any(word in content for word in ["very good", "happy", "great", "appreciate", "excellent"]):
        sentiment = "positive"
    elif any(word in content for word in ["angry", "terrible", "terrible", "damn"]):
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Category classification (very naive)
    if any(word in content for word in ["support", "help", "issue"]):
        category = "support"
    elif any(word in content for word in ["billing", "invoice", "payment"]):
        category = "billing"
    elif any(word in content for word in ["sales", "buy", "purchase"]):
        category = "sales"
    else:
        category = "general"

    print(f"Classified: {priority} priority, {sentiment} sentiment, {category} category")

    return {
        **state, 
        "priority": priority, 
        "sentiment": sentiment, 
        "category": category
    }


def extract_entities(state: EmailState) -> EmailState:
    """Extract emails, phone numbers, and dates"""
    content = state["email_content"]

    emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', content)
    phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b', content)
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', content)

    entities = emails + phones + dates

    print(f"  Extracted {len(entities)} entities")

    return {
        **state, 
        "entities": entities
    }


def route_email(state: EmailState) -> EmailState:
    """Determine routing based on classification"""
    priority = state["priority"]
    category = state["category"]
    sentiment = state["sentiment"]

    # Routing logic
    if priority == "high" or sentiment == "negative":
        route = "escalate"
    elif category == "support":
        route = "support_team"
    elif category == "billing":
        route = "billing_team"
    elif category == "sales":
        route = "sales_team"
    else:
        route = "general_queue"

    print(f"  Routed to: {route}")

    return {
        **state, 
        "route": route
    }


def generate_reply(state: EmailState) -> EmailState:
    """Generate auto-reply based on analysis"""
    category = state["category"]
    priority = state["priority"]
    sentiment = state["sentiment"]

    # Generate appropriate reply
    if sentiment == "negative":
        reply = f"We sincerely apologize for any inconvenience. Your {category} concern is our priority and has been escalated."
    elif category == "support":
        reply = "Thank you for contacting support. We've received your request and will respond within 24 hours."
    elif category == "billing":
        reply = "Your billing inquiry has been forwarded to our accounts team. Expect a response within 2 business days."
    elif category == "sales":
        reply = "Thank you for your interest! Our sales team will contact you shortly with more information."
    else:
        reply = "Thank you for your message. We've received it and will respond appropriately."

    if priority == "high":
        reply += " Due to the urgent nature, this will be prioritized."

    print(f"  Reply generated")

    return {
        **state, 
        "reply": reply
    }
 


workflow = StateGraph(EmailState)

workflow.add_node("classifier", classify_email)
workflow.add_node("entity_extractor", extract_entities)
workflow.add_node("router", route_email)
workflow.add_node("reply_generator", generate_reply)

workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "entity_extractor")
workflow.add_edge("entity_extractor", "router")
workflow.add_edge("router", "reply_generator")
workflow.add_edge("reply_generator", END)

email_graph = workflow.compile()


def test_email_system():
    test_email = [
                {
            "email_content": "URGENT: My account is locked and I can't access my files! This is terrible!",
            "sender": "angry.customer@email.com"
        },
        {
            "email_content": "Hi, I'm interested in purchasing your premium plan. Can you send me pricing info?",
            "sender": "potential.buyer@company.com"
        },
        {
            "email_content": "I have a question about my recent invoice. Please contact me at 555-123-4567.",
            "sender": "customer@domain.com"
        },
    ]

    for i, email in enumerate(test_email, 1):
        print(f"\nEmail {i}: {email['sender']}")
        print(f"Content: {email['email_content'][:50]}...")

        # Process the email
        result = email_graph.invoke({
            "email_content": email["email_content"],
            "sender": email["sender"],
            "priority": "",
            "sentiment": "",
            "category": "",
            "entities": [],
            "route": "",
            "reply": ""
        })

        print(f"\nResults:")
        print(f"  Priority: {result['priority']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Category: {result['category']}")
        print(f"  Route: {result['route']}")
        print(f"  Entities: {len(result['entities'])} found")
        print(f"  Reply: {result['reply']}")
        print("-" * 40)


if __name__ == "__main__":
    test_email_system()
