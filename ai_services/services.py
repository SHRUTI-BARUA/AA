import os
import google.generativeai as genai
import json
from typing import List, Dict, Any
from models import MatchProfile
import requests

# Configure Gemini
# In production, ensure GEMINI_API_KEY is set in environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY, transport='rest')

def get_gemini_model():
    return genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

GOVT_KEYWORDS = [
    "shree anna", "millet mission", "official scheme", "government benefits", 
    "msp", "pmmsy", "subsidy", "policy", "yojana"
]

async def generate_chat_response(query: str, context: str = "") -> Dict[str, Any]:
    try:
        # 0. Handle Greeting specifically
        if "namaste" in query.lower() or "hello" in query.lower() and len(query.split()) < 5:
             return {
                "answer": "Namaste! I am your Millet Assistant. Ask me anything.",
                "sources": []
            }

        # 1. Intent Detection for Web Search (Up-to-date data)
        # Expanded keywords for real-time needs
        SEARCH_KEYWORDS = [
            "shree anna", "millet mission", "official scheme", "government benefits", 
            "msp", "pmmsy", "subsidy", "policy", "yojana",
            "price", "rate", "market", "today", "latest", "news", "current"
        ]
        
        needs_web_search = any(keyword in query.lower() for keyword in SEARCH_KEYWORDS)
        
        if needs_web_search:
            # Web Search Mode via Direct REST API (to bypass SDK version issues)
            api_key = os.getenv("GEMINI_API_KEY")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
            
            prompt = f"""
            You are an AI assistant for Indian farmers.
            User Query: {query}
            
            Task:
            1. Search official government websites or news sources for this query.
            2. Extract verified, up-to-date information.
            3. Answer in simple, farmer-friendly English.
            4. List the sources used.
            
            Return a JSON object with:
            - answer: The explanation.
            - sources: List of URLs or source names.
            """
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "tools": [{
                    "google_search": {}
                }]
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code != 200:
                return {
                    "answer": f"Error from Google API: {response.text}",
                    "sources": []
                }
                
            data = response.json()
            
            # Extract text and sources
            try:
                # Gemini 2.0 structure might vary, but usually candidates[0].content.parts[0].text
                candidate = data.get("candidates", [])[0]
                text_part = candidate.get("content", {}).get("parts", [])[0].get("text", "")
                
                # Extract grounding metadata for sources if available
                grounding_metadata = candidate.get("groundingMetadata", {})
                chunks = grounding_metadata.get("groundingChunks", [])
                sources = [chunk.get("web", {}).get("uri") for chunk in chunks if "web" in chunk]
                
                # Try to parse JSON from text
                clean_text = text_part.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(clean_text)
                except:
                     # If not JSON, just return text
                    result = {"answer": clean_text, "sources": sources}
                
                # Merge sources if not present in JSON
                if not result.get("sources") and sources:
                    result["sources"] = sources
                    
                return result
            except Exception as parse_error:
                # Fallback
                return {
                    "answer": text_part if 'text_part' in locals() else "Could not parse response",
                    "sources": sources if 'sources' in locals() else ["Google Search"]
                }
                
        else:
            # Fast Path: General Questions (No Web Search)
            # Use a faster model or just standard generation without tools
            model = get_gemini_model()
            prompt = f"""
            You are a helpful Millet Assistant. Answer the following question quickly and concisely.
            Context: {context}
            User: {query}
            Assistant:
            """
            response = model.generate_content(prompt)
            return {
                "answer": response.text,
                "sources": []
            }
            
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "sources": []
        }

async def translate_text(text: str, target_lang: str) -> str:
    try:
        model = get_gemini_model()
        prompt = f"Translate the following text to {target_lang}. Return ONLY the translated text, no explanations.\n\nText: {text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error translating: {str(e)}"

async def get_market_price(millet_type: str, quality_grade: str, location: str) -> Dict:
    try:
        # Use simple model logic if no API key for search, but preferred to be real-time
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             # Fallback to simple estimation if key missing
             return {
                "market_price": 2500,
                "recommended_price": 2700,
                "currency": "INR",
                "reasoning": "API Key missing, returning estimation."
            }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
            
        prompt = f"""
        Act as an agricultural market expert in India.
        Task:
        1. Search for the CURRENT (last 7 days) market price (mandi price) of {millet_type} in {location} or nearby markets in India.
        2. Establish a realistic Base Price per Quintal (100kg).
        
        Then apply these rules for the Recommended Price:
        - If Quality A (Premium): Price = Base Price + 10%
        - If Quality B (Standard): Price = Base Price
        - If Quality C (Fair): Price = Base Price - 10%
        
        Current Quality: {quality_grade}
        
        Return a JSON object with:
        - market_price (numeric, base price per Quintal)
        - recommended_price (numeric, calculated price per Quintal)
        - currency (string, e.g., "INR")
        - reasoning (string, brief explanation with source if possible, e.g. "Based on average mandi prices in Rajasthan...")
        
        Return ONLY JSON.
        """
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "tools": [{
                "google_search": {}
            }]
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
             # Fallback
             print(f"API Error: {response.text}") # Debug
             return {"market_price": 0, "recommended_price": 0, "currency": "INR", "reasoning": "Search API Failed"}

        data = response.json()
        
        try:
            candidate = data.get("candidates", [])[0]
            text_part = candidate.get("content", {}).get("parts", [])[0].get("text", "")
            
            # Clean up response to ensure it's valid JSON
            clean_text = text_part.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as parse_error:
            # Try to extract numbers if JSON fails
            print(f"JSON Parse Error: {parse_error}")
            return {
                "market_price": 0,
                "recommended_price": 0,
                "currency": "INR", 
                "reasoning": f"Could not parse AI response: {text_part[:100]}"
            }
            
    except Exception as e:
        return {
            "market_price": 0,
            "recommended_price": 0,
            "currency": "INR",
            "reasoning": f"Failed to fetch price: {str(e)}"
        }

async def analyze_quality(millet_type: str, description: str, impurities: str) -> Dict:
    try:
        model = get_gemini_model()
        prompt = f"""
        Analyze the quality of this millet sample based on the description.
        
        Millet Type: {millet_type}
        Description: {description}
        Impurities: {impurities}
        
        Return a JSON object with:
        - qualityGrade (A, B, or C)
        - moistureEstimate (e.g., "10-12%")
        - cleanliness (High/Medium/Low)
        - adulterationRisk (Low/Medium/High)
        - recommendation (Brief advice)
        
        Return ONLY JSON.
        """
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {
            "qualityGrade": "Unknown",
            "moistureEstimate": "Unknown",
            "cleanliness": "Unknown",
            "adulterationRisk": "Unknown",
            "observedIssues": [],
            "recommendation": f"Error: {str(e)}"
        }

async def analyze_quality_image(millet_type: str, image_bytes: bytes) -> Dict:
    try:
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        model = get_gemini_model()
        
        prompt = f"""
        Analyze the quality of this {millet_type} sample based on the image.
        Evaluate:
        1. Color and visual appearance (brightness, discoloration)
        2. Grain uniformity (size, shape)
        3. Visible impurities (dust, stones, husk, foreign matter)
        4. Dryness (visual estimate of moisture)
        5. Signs of adulteration or pest damage
        
        Return a JSON object with:
        - qualityGrade (A, B, or C)
        - moistureEstimate (e.g., "10-12%")
        - cleanliness (High/Medium/Low)
        - adulterationRisk (Low/Medium/High)
        - observedIssues (List of specific visual defects, e.g., ["excessive husk", "discolored grains", "stone particles"])
        - recommendation (Specific, actionable advice for processing or storage based on the observed issues. Avoid generic statements.)
        
        Return ONLY JSON.
        """
        
        response = model.generate_content([prompt, image])
        
        try:
            # Clean up response to ensure it's valid JSON
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as parse_error:
            print(f"JSON Parse Error: {parse_error}")
            print(f"Raw Response: {response.text}")
            raise parse_error
            
    except Exception as e:
        return {
            "qualityGrade": "Unknown",
            "moistureEstimate": "Unknown",
            "cleanliness": "Unknown",
            "adulterationRisk": "Unknown",
            "observedIssues": ["Error processing image"],
            "recommendation": f"Error: {str(e)}"
        }

# Mock Data for Matching
# Mock Data for Matching
MOCK_FARMERS = [
    MatchProfile(id="f1", name="Ramesh Kumar", type="farmer", millet_type="Pearl Millet", quantity=100, location="Rajasthan"),
    MatchProfile(id="f2", name="Suresh Singh", type="farmer", millet_type="Sorghum", quantity=500, location="Maharashtra"),
    MatchProfile(id="f3", name="Anita Devi", type="farmer", millet_type="Finger Millet", quantity=200, location="Karnataka"),
    MatchProfile(id="f4", name="Rajesh Gupta", type="farmer", millet_type="Foxtail Millet", quantity=150, location="Andhra Pradesh"),
    MatchProfile(id="f5", name="Vikram Singh", type="farmer", millet_type="Pearl Millet", quantity=300, location="Gujarat"),
]

MOCK_BUYERS = [
    MatchProfile(id="b1", name="Millet Foods Ltd", type="buyer", millet_type="Pearl Millet", quantity=1000, location="Delhi"),
    MatchProfile(id="b2", name="Healthy Grains", type="buyer", millet_type="Finger Millet", quantity=150, location="Bangalore"),
]

async def match_users(user_type: str, millet_type: str, quantity: float, location: str) -> List[MatchProfile]:
    try:
        # Get Candidates (In a real app, this comes from DB)
        candidates = MOCK_BUYERS if user_type == "farmer" else MOCK_FARMERS
        
        # Prepare content for Gemini to Rank/Filter
        candidates_json = json.dumps([c.__dict__ for c in candidates])
        
        model = get_gemini_model()
        prompt = f"""
        Act as a B2B agricultural matching engine.
        
        My Profile:
        - Role: {user_type}
        - Selling/Buying: {millet_type}
        - Quantity: {quantity} kg
        - Location: {location}
        
        Potential Matches:
        {candidates_json}
        
        Task:
        1. Analyze each candidate to find the best trading partners.
        2. Score them from 0-100 based on:
           - Millet Type match (Strict)
           - Location proximity (Preferred)
           - Quantity capacity (Can they handle the order?)
        3. Return ONLY the candidates that have a score > 50.
        4. Sort by score descending.
        
        Return JSON array of objects (structure same as input but can have 'match_score' field).
        Return ONLY JSON.
        """
        
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        
        try:
            matches_data = json.loads(text)
            # Convert back to MatchProfile objects
            matches = []
            for m in matches_data:
                # Ensure it matches MatchProfile structure
                matches.append(MatchProfile(
                    id=m.get("id"),
                    name=m.get("name"), 
                    type=m.get("type"),
                    millet_type=m.get("millet_type"),
                    quantity=m.get("quantity"),
                    location=m.get("location")
                ))
            return matches
        except:
             # Fallback to simple logic if AI fails
             return [c for c in candidates if millet_type.lower() in c.millet_type.lower()]
             
    except Exception as e:
        print(f"Matching error: {e}")
        return []

async def get_market_trends(millet_type: str) -> List[Dict[str, Any]]:
    try:
        # Web Search Mode via Direct REST API
        api_key = os.getenv("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
        
        prompt = f"""
        Act as a market data analyst.
        Task: Search for the daily wholesale market prices (per quintal) of {millet_type} in major Indian mandis (markets) for the last 15-30 days.
        
        Return a JSON array of objects. Each object must have:
        - date: "YYYY-MM-DD" format
        - price_per_quintal: number (in INR)
        - market_name: string (e.g., "Jaipur Mandi", "Nizamabad")
        
        Rules:
        1. Try to find real recent data points.
        2. If exact daily data is missing, estimate based on weekly trends or news reports, but mark them as estimates.
        3. Return at least 10-15 data points sorted by date.
        4. Return ONLY the JSON array.
        """
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "tools": [{
                "google_search": {}
            }]
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            print(f"Error from Google API: {response.text}")
            return []
            
        data = response.json()
        
        # Extract text
        try:
            candidate = data.get("candidates", [])[0]
            text_part = candidate.get("content", {}).get("parts", [])[0].get("text", "")
            
            # Clean up JSON
            clean_text = text_part.replace("```json", "").replace("```", "").strip()
            # Find the start and end of the list
            start = clean_text.find("[")
            end = clean_text.rfind("]") + 1
            if start != -1 and end != -1:
                clean_text = clean_text[start:end]
                
            return json.loads(clean_text)
        except Exception as parse_error:
            print(f"Parsing error: {parse_error}")
            return []
            
    except Exception as e:
        print(f"Error fetching market trends: {str(e)}")
        return []
