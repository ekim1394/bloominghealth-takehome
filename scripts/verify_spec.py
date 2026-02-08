import asyncio
import httpx
import json
import os
from typing import Any

BASE_URL = "http://127.0.0.1:8013/api"

SPEC_CASES = [
    {
        "id": "eval_001",
        "context": {
            "directive": "Verify the caller's date of birth",
            "user_input": "It's March 15th, 1985"
        },
        "response_good": "Got it, March 15th, 1985. Thank you for confirming that.",
        "response_bad": "Perfect! I have recorded your date of birth as March 15th, 1985. Is there anything else I can help you verify today?",
        "expected_winner": "response_good",
        "description": "DOB Verification (Conciseness)"
    },
    {
        "id": "eval_002",
        "context": {
            "directive": "Ask about food security using SDOH screening question",
            "user_input": "We haven't had enough to eat in weeks. It's been really hard."
        },
        "response_good": "I'm sorry to hear that - that sounds really difficult. I want to make sure we can connect you with the right resources. Within the past 12 months, did you worry that your food would run out before you got money to buy more?",
        "response_bad": "Okay, noted. Next question: Within the past 12 months, did you worry that your food would run out before you got money to buy more?",
        "expected_winner": "response_good",
        "description": "Food Security (Empathy)"
    },
    {
        "id": "eval_003",
        "context": {
            "directive": "Handle user confusion about the survey purpose",
            "user_input": "Wait, why are you asking me all these personal questions? Who is this for?"
        },
        "response_good": "That's a fair question. This survey helps us understand if there are any areas where you might benefit from community resources or extra support. It's completely optional, but it helps us provide better care for you.",
        "response_bad": "I'm an AI assistant helping with your health screening. These questions are part of a standard SDOH assessment used by healthcare providers to track social determinants of health.",
        "expected_winner": "response_good",
        "description": "User Confusion (Naturalness)"
    }
]

async def test_evaluate(client: httpx.AsyncClient, case: dict[str, Any], type: str):
    response_text = case[f"response_{type}"]
    print(f"\n[EVALUATE] {case['id']} ({type}): {response_text[:60]}...")
    
    payload = {
        "user_input": case["context"]["user_input"],
        "context": case["context"]["directive"],
        "response": response_text
    }
    
    resp = await client.post("/evaluate", json=payload)
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return None
    
    result = resp.json()
    print(f"  Overall Score: {result['overall_score']}")
    print(f"  Flags: {result['flags']}")
    return result

async def test_compare(client: httpx.AsyncClient, case: dict[str, Any]):
    print(f"\n[COMPARE] {case['id']}: {case['description']}")
    
    payload = {
        "user_input": case["context"]["user_input"],
        "context": case["context"]["directive"],
        "response_a": case["response_good"],
        "response_b": case["response_bad"]
    }
    
    resp = await client.post("/compare", json=payload)
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return
    
    result = resp.json()
    winner = result["winner"]
    print(f"  Winner: {winner} (Expected: A)")
    print(f"  Reasoning: {result['reasoning']}")
    return result

async def test_batch(client: httpx.AsyncClient):
    print("\n[BATCH EVALUATE]")
    items = []
    for case in SPEC_CASES:
        items.append({
            "user_input": case["context"]["user_input"],
            "context": case["context"]["directive"],
            "response": case["response_good"]
        })
    
    resp = await client.post("/evaluate/batch", json={"items": items})
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return
    
    result = resp.json()
    print(f"  Overall Average: {result['overall_average']}")
    print(f"  Total Flags: {result['total_flags']}")
    print(f"  Aggregate Scores: {json.dumps(result['aggregate'], indent=2)}")

async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # Check health/wait for server
        try:
            await client.get("http://localhost:8000/health")
        except Exception:
            print("Error: Server not running at http://localhost:8000")
            return

        print("Starting Case Study 3 Specification Tests...")
        
        # 1. Test batch evaluation
        await test_batch(client)
        
        # 2. Run through each case for comparison
        for case in SPEC_CASES:
            await test_compare(client, case)
            # Short delay between runs
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
