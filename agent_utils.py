from google import genai
import os
import json
import time

# Use environment variable for security
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------------------------------
# 1️⃣ STYLE + ATTRIBUTE EXTRACTION
# ------------------------------------------------
def extract_style(query):

    prompt = f"""
    Analyze the following fashion request:
    "{query}"

    Extract structured information in STRICT JSON format:

    {{
        "occasion": "",
        "style": "",
        "color_preference": "",
        "mood": "",
        "season": "",
        "formality_level": "",
        "gender_target": ""
    }}

    Return only valid JSON.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        return json.loads(response.text)
    except:
        return {}


# ------------------------------------------------
# 2️⃣ INTENT CLASSIFICATION
# ------------------------------------------------
def classify_intent(query):

    prompt = f"""
    Classify the intent of this fashion query:
    "{query}"

    Choose one:
    - outfit_recommendation
    - style_inspiration
    - color_advice
    - trend_inquiry
    - image_matching
    - general_fashion_question

    Return only the category name.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# ------------------------------------------------
# 3️⃣ CONSTRAINT DETECTION
# ------------------------------------------------
def detect_constraints(query):

    prompt = f"""
    Identify if the query includes constraints such as:
    - budget
    - season
    - weather
    - cultural requirements
    - dress code

    Return them in JSON format:
    {{
        "budget": "",
        "weather": "",
        "cultural": "",
        "dress_code": ""
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        return json.loads(response.text)
    except:
        return {}


# ------------------------------------------------
# 4️⃣ PERSONALIZED EXPLANATION
# ------------------------------------------------
def generate_explanation(query, intent, attributes, constraints, results):

    top_items = "\n".join(results[:5])

    prompt = f"""
    You are a professional AI Fashion Stylist helping a client choose outfits.

    User Request:
    {query}

    Detected Intent:
    {intent}

    Extracted Style Attributes:
    {attributes}

    User Constraints:
    {constraints}

    Recommended Outfit Items:
    {top_items}

    Explain in detail why these recommended outfits match the user's request.

    Your explanation should include:

    1. Style reasoning – how the outfit types match the request
    2. Color coordination – why the colors work well
    3. Occasion suitability – why these outfits are appropriate
    4. Fashion advice – suggest accessories or styling tips
    5. Overall aesthetic – describe the look and feel of the outfits

    Write like a professional stylist advising a client.
    Write 150–200 words.
    Do NOT repeat JSON attributes.
    Return only the explanation text.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text




# ------------------------------------------------
# 6️⃣ LOGGING THE INTERACTION
# ------------------------------------------------
def log_interaction(query, intent, attributes, constraints, explanation):

    with open("agent_logs.txt", "a") as f:
        f.write(f"\nTime: {time.ctime()}")
        f.write(f"\nQuery: {query}")
        f.write(f"\nIntent: {intent}")
        f.write(f"\nAttributes: {attributes}")
        f.write(f"\nConstraints: {constraints}")
        f.write(f"\nExplanation: {explanation}")
        f.write("\n" + "-"*60)
        



# -------------------------------------------------
# AI OUTFIT COMPLETER UTILITIES
# -------------------------------------------------

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)


def detect_category_from_image(image):

    categories = [
        "shirt",
        "t-shirt",
        "dress",
        "jacket",
        "pants",
        "jeans",
        "skirt",
        "shoes",
        "sneakers",
        "bag"
    ]

    image_input = preprocess(image).unsqueeze(0).to(device)

    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in categories]
    ).to(device)

    with torch.no_grad():

        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

        logits = (image_features @ text_features.T).softmax(dim=-1)

    predicted_index = logits.argmax().item()

    return categories[predicted_index]


def normalize_category(category):

    mapping = {

        "shirt": "top",
        "t-shirt": "top",
        "jacket": "outerwear",

        "pants": "bottom",
        "jeans": "bottom",
        "skirt": "bottom",

        "dress": "dress",

        "shoes": "shoes",
        "sneakers": "shoes",

        "bag": "bag"
    }

    return mapping.get(category, category)


def complementary_categories(category):

    rules = {

        "top": ["bottom", "shoes", "bag"],

        "bottom": ["top", "shoes", "outerwear"],

        "dress": ["shoes", "bag"],

        "outerwear": ["top", "bottom", "shoes"]
    }

    return rules.get(category, [])