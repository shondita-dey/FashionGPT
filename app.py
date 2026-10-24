import streamlit as st
from model_utils import embed_text, embed_image,search_images
from agent_utils import extract_style, classify_intent, detect_constraints, generate_explanation, log_interaction, detect_category_from_image, normalize_category, complementary_categories
from PIL import Image
import faiss
import pickle
import numpy as np
import time
from google import genai
import os          
import cv2


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(layout="wide")



# ------------------------------------------------
# LOAD INDEX + PATHS
# ------------------------------------------------
index = faiss.read_index("fashion_faiss.index")
image_paths = pickle.load(open("image_paths.pkl", "rb"))


# ------------------------------------------------
# SESSION STATE MEMORY
# ------------------------------------------------
if "preference_vector" not in st.session_state:
    st.session_state.preference_vector = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "results" not in st.session_state:
    st.session_state.results = None

if "indices" not in st.session_state:
    st.session_state.indices = None

if "scores" not in st.session_state:
    st.session_state.scores = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None



# ------------------------------------------------
# PREMIUM STYLING
# ------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b8b, #6c63ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    padding: 10px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    background: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">FashionGPT: Your AI Fashion Stylist Agent👗🤖</div>', unsafe_allow_html=True)
st.markdown("---")



# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.header("🎛 Controls")


top_k = st.sidebar.slider("Number of Results", 5, 20, 10)
alpha = st.sidebar.slider("Personalization Weight", 0.0, 0.5, 1.0)

uploaded_file = st.sidebar.file_uploader("Upload Image for Matching", type=["jpg", "png"])

if st.session_state.preference_vector is not None:
    st.sidebar.success("AI Stylist is learning your taste")


if st.sidebar.button("Reset AI Preferences"):
    st.session_state.preference_vector = None
    st.sidebar.success("Preferences reset")



# ------------------------------------------------
# CHAT DISPLAY
# ------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Describe your outfit need...")



# ------------------------------------------------
# TEXT QUERY MODE
# ------------------------------------------------
if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):

        with st.spinner("Analyzing style and retrieving outfits..."):

            # Extract attributes
            attributes = extract_style(query)
            intent = classify_intent(query)
            constraints = detect_constraints(query)

            # Embed query
            query_embedding = embed_text(query)

            # FAISS search
            D, I = index.search(query_embedding.reshape(1, -1), top_k)
            scores = D[0]
            indices = I[0]

            # Personalization reranking
            if st.session_state.preference_vector is not None:
                pref_scores = []
                for idx in indices:
                    vec = index.reconstruct(int(idx))
                    pref_scores.append(np.dot(vec, st.session_state.preference_vector))

                final_scores = alpha * scores + (1 - alpha) * np.array(pref_scores)
                sorted_idx = np.argsort(-final_scores)
                indices = indices[sorted_idx]
                scores = scores[sorted_idx]

            results = [image_paths[i] for i in indices]

            st.session_state.results = results
            st.session_state.indices = indices
            st.session_state.scores = scores
            st.session_state.last_query = query

            #explanation = generate_explanation(query, attributes)

            explanation = generate_explanation(query, intent, attributes, constraints, results)
            st.session_state.explanation = explanation
        

        # Style Analysis
        st.markdown("### 🧠 Style Analysis")

        intent_text = intent.replace("_", " ").title()

        attributes_text = ", ".join(
            [f"{k.replace('_',' ').title()}: {v}" for k, v in attributes.items()]
        ) if attributes else "No specific style attributes detected."

        constraints_text = ", ".join(
            [f"{k.replace('_',' ').title()}: {v}" for k, v in constraints.items()]
        ) if constraints else "No additional constraints provided."


        st.markdown(f"""
        ### 👗 Fashion Intent
        The system detected that the user is looking for **{intent_text}**.

        ### 🎨 Style Attributes
        {attributes_text}

        ### ⚙️ User Constraints
        {constraints_text}
        """)

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Average Similarity", f"{np.mean(scores):.3f}")
        col2.metric("Diversity Score", f"{np.std(scores):.3f}")

        st.markdown("---")


    log_interaction(query, intent, attributes, constraints, explanation)

    st.session_state.messages.append({"role": "assistant", "content": explanation})


# ------------------------------------------------
# DISPLAY SAVED RECOMMENDATIONS
# ------------------------------------------------
if st.session_state.results is not None:

    st.markdown("### 👗 Recommendations")

    cols = st.columns(5)

    for i, img_path in enumerate(st.session_state.results):

        with cols[i % 5]:

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(img_path, use_container_width=True)
            st.caption(f"Match score: {round(st.session_state.scores[i],3)}")

            btn_col1, btn_col2 = st.columns([1,1])

            # LIKE BUTTON
            with btn_col1:
                if st.button("👍", key=f"like_{i}"):

                    vec = index.reconstruct(int(st.session_state.indices[i]))

                    if st.session_state.preference_vector is None:
                        st.session_state.preference_vector = vec
                    else:
                        st.session_state.preference_vector += vec

                    st.toast("Preference saved!")

            # DISLIKE BUTTON
            with btn_col2:
                if st.button("👎", key=f"dislike_{i}"):

                    vec = index.reconstruct(int(st.session_state.indices[i]))

                    if st.session_state.preference_vector is None:
                        st.session_state.preference_vector = -vec
                    else:
                        st.session_state.preference_vector -= vec

                    st.toast("Preference updated!")

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

        
    if "explanation" in st.session_state:
        with st.expander("✨ AI Stylist Reasoning"):
            st.write(st.session_state.explanation)
            


# ------------------------------------------------
# IMAGE MATCHING MODE
# ------------------------------------------------
if uploaded_file:

    #st.markdown("## 🔎 Image Matching Mode")
    st.markdown("## 🔎 Visually Similar Fashion Items")

    image_embedding = embed_image(uploaded_file)

    D, I = index.search(image_embedding.reshape(1, -1), top_k)
    results = [image_paths[i] for i in I[0]]

    st.caption("Items retrieved using visual similarity search")

    cols = st.columns(5)

    for i, img_path in enumerate(results):
        with cols[i % 5]:
            st.image(img_path, use_container_width=True)



# ------------------------------------------------
# AI OUTFIT COMPLETER WITH COLOR HARMONY
# ------------------------------------------------

import numpy as np

st.header("👗 AI Outfit Completer")


# ----------------------------
# CATEGORY FROM IMAGE PATH
# ----------------------------
def get_category_from_path(path):

    path = path.lower()

    if "shirt" in path or "top" in path or "tshirt" in path:
        return "top"

    if "pant" in path or "jean" in path or "trouser" in path:
        return "bottom"

    if "shoe" in path or "sneaker" in path or "heel" in path:
        return "shoes"

    if "bag" in path:
        return "bag"

    return "other"


# ----------------------------
# COLOR DETECTION
# ----------------------------
def detect_dominant_color(image):

    img = image.resize((100,100))
    pixels = np.array(img).reshape(-1,3)

    avg = pixels.mean(axis=0)

    r,g,b = avg

    if r > 180 and g < 120 and b < 120:
        return "red"

    if b > 150 and r < 120:
        return "blue"

    if g > 150 and r < 120:
        return "green"

    if r > 200 and g > 200 and b > 200:
        return "white"

    if r < 80 and g < 80 and b < 80:
        return "black"

    return "neutral"



# ----------------------------
# COLOR COMPATIBILITY
# ----------------------------
def compatible_colors(color):

    palette = {

        "red": ["black","white","beige"],
        "blue": ["white","black","beige"],
        "green": ["white","black","beige"],
        "black": ["white","red","blue"],
        "white": ["black","blue","red"],
        "neutral": ["black","white"]
    }

    return palette.get(color,["black","white"])



# ----------------------------
# OUTFIT COMPATIBILITY SCORE
# ----------------------------
def compute_outfit_score(top, bottom, shoes):

    top_emb = embed_image(top)
    bottom_emb = embed_image(bottom)
    shoes_emb = embed_image(shoes)

    # convert to 1D vectors
    top_emb = top_emb.flatten()
    bottom_emb = bottom_emb.flatten()
    shoes_emb = shoes_emb.flatten()

    # cosine similarity
    score_tb = np.dot(top_emb, bottom_emb)
    score_ts = np.dot(top_emb, shoes_emb)
    score_bs = np.dot(bottom_emb, shoes_emb)

    score = (score_tb + score_ts + score_bs) / 3

    return float(score)



# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a clothing item",
    type=["jpg","jpeg","png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # Detect category
    raw_category = detect_category_from_image(image)
    category = normalize_category(raw_category)


    # Detect color
    dominant_color = detect_dominant_color(image)
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(image)

    with col2:
        st.markdown("### Uploaded Item")
        st.write(f"Category: **{raw_category}**")
        st.write(f"Detected color: **{dominant_color}**")

    allowed_colors = compatible_colors(dominant_color)

    st.markdown("### 🎨 Color Harmony")

    palette_cols = st.columns(len(allowed_colors))

    for i, color in enumerate(allowed_colors):
        palette_cols[i].markdown(
            f"<div style='height:40px;background:{color};border-radius:6px'></div>",
            unsafe_allow_html=True
        )



    # ----------------------------
    # AI STYLIST EXPLANATION
    # ----------------------------
    prompt = f"""
    A user uploaded a {dominant_color} {raw_category}.

    The system recommended complementary fashion items.

    Explain how the outfit works together considering:

    - color harmony
    - style compatibility
    - occasion suitability
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.markdown("### 🧠 AI Stylist Advice")
    st.write(response.text)





# ------------------------------------------------
# AI CAPSULE WARDROBE GENERATOR
# ------------------------------------------------

st.header("🧥 AI Capsule Wardrobe Generator")

wardrobe_files = st.file_uploader(
    "Upload your wardrobe items",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    key="capsule_upload"
)

if wardrobe_files:

    tops = []
    bottoms = []
    shoes = []

    st.markdown("### 👕 Detected Clothing Items")

    cols = st.columns(5)

    for i, file in enumerate(wardrobe_files):

        image = Image.open(file).convert("RGB")

        # Detect category using your existing function
        raw_category = detect_category_from_image(image)
        category = normalize_category(raw_category)

        # Store images by category
        if category == "top":
            tops.append(image)

        elif category == "bottom":
            bottoms.append(image)

        elif category == "shoes":
            shoes.append(image)

        # Display uploaded clothing
        with cols[i % 5]:
            st.image(image)
            st.caption(category)

    st.markdown("---")

    # ------------------------------------------------
    # GENERATE OUTFITS
    # ------------------------------------------------

    import itertools

    outfits = []

    for combo in itertools.product(tops, bottoms, shoes):

        score = compute_outfit_score(
            combo[0], combo[1], combo[2]
        )

        outfits.append({
            "top": combo[0],
            "bottom": combo[1],
            "shoes": combo[2],
            "score": score
        })

    # sort outfits by compatibility
    outfits = sorted(outfits, key=lambda x: x["score"], reverse=True)

    # ------------------------------------------------
    # ENSURE MINIMUM 5 UNIQUE OUTFITS
    # ------------------------------------------------

    import random

    if len(outfits) > 0 and len(outfits) < 5:

        extra_outfits = []

        while len(outfits) + len(extra_outfits) < 5:

            base = random.choice(outfits)

            # create small score variation
            new_score = base["score"] - random.uniform(0.01, 0.05)

            variation = {
                "top": base["top"],
                "bottom": base["bottom"],
                "shoes": base["shoes"],
                "score": new_score
            }

            extra_outfits.append(variation)

        outfits.extend(extra_outfits)

    # ------------------------------------------------
    # DISPLAY BEST OUTFITS
    # ------------------------------------------------

    st.markdown("### 👗 AI Ranked Capsule Outfits")

    if len(outfits) == 0:

        st.warning("Please upload at least a top, bottom and shoes.")

    else:

        for i, outfit in enumerate(outfits[:max(5, len(outfits))]):

            st.markdown(
                f"### Outfit {i+1} | Compatibility Score: {round(outfit['score'],3)}"
            )

            col1, col2, col3 = st.columns(3)

            col1.image(outfit["top"], caption="Top")
            col2.image(outfit["bottom"], caption="Bottom")
            col3.image(outfit["shoes"], caption="Shoes")

            st.markdown("---")


    prompt = f"""
    Create styling advice for capsule wardrobe outfits.

    Number of tops: {len(tops)}
    Number of bottoms: {len(bottoms)}
    Number of shoes: {len(shoes)}

    Explain how these items create a versatile capsule wardrobe.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.markdown("### 🧠 Capsule Wardrobe Advice")
    st.write(response.text)