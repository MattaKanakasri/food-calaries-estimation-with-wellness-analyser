# -------------------
# 🍔 Food Pro App — Food Calories Estimation & Wellness Analyzer
# -------------------

import pandas as pd
import pickle
import cv2
import numpy as np
import os
from difflib import get_close_matches
import streamlit as st
from skimage.feature import hog
import random
import datetime
import time
from PIL import Image

# -------------------
# Load model & data
# -------------------
with open("train_model.pkl", "rb") as f:
    model = pickle.load(f)

recipes = pd.read_csv("recipes.csv")
recipes.rename(columns={
    recipes.columns[0]: "food_name",
    "ingredients": "ingredients",
    "calories": "calories",
    "Healthy substitute": "substitute"
}, inplace=True)

# -------------------
# Prediction & recipe helpers
# -------------------
def predict_food(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Unknown"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    features = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    features = features.reshape(1, -1)
    try:
        return model.predict(features)[0]
    except:
        return "Unknown"

def get_recipe(food_name):
    all_foods = recipes["food_name"].tolist()
    food_name_clean = food_name.lower().strip()
    all_foods_clean = [f.lower().strip() for f in all_foods]

    match = get_close_matches(food_name_clean, all_foods_clean, n=1, cutoff=0.3)
    if match:
        idx = all_foods_clean.index(match[0])
        recipe = recipes.iloc[[idx]]
    else:
        recipe = recipes.sample(1)

    ingredients = recipe["ingredients"].values[0]
    calories = recipe["calories"].values[0]
    substitute = (
        recipe["substitute"].values[0]
        if "substitute" in recipe.columns and recipe["substitute"].values[0]
        else "Use healthier ingredients: reduce sugar, use whole grains 🍏"
    )
    return ingredients, calories, substitute

def similar_recipes(food_name, n=3):
    all_foods = recipes["food_name"].tolist()
    matches = get_close_matches(food_name, all_foods, n=n, cutoff=0.5)
    matches = [m for m in matches if m != food_name]
    if not matches:
        matches = random.sample(all_foods, min(n, len(all_foods)))
    return matches

def estimate_calories_from_image(img_path, predicted_food):
    recipe = recipes[recipes["food_name"] == predicted_food]
    if not recipe.empty and "calories" in recipe.columns:
        try:
            return float(recipe["calories"].values[0])
        except:
            return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    avg_brightness = np.mean(img)
    img_area = img.shape[0] * img.shape[1]
    approx_calories = round((avg_brightness / 255) * (img_area / 10000) * 50, 2)
    return approx_calories if approx_calories > 0 else None

# -------------------
# UI Setup
# -------------------
st.set_page_config(page_title="🍔 Food Pro App", layout="wide")

st.markdown("""
<style>
body {background-color: #fdf6f0;}
.stSidebar {background-color: #ffd9b3; padding: 15px; border-radius: 10px;}
.header {text-align:center; color:#ff5c4d; font-size:42px; font-weight:bold; margin-bottom:20px;}
.card {background-color:#fff9f2; border-radius:15px; padding:15px;
       box-shadow:2px 2px 20px rgba(0,0,0,0.15); margin-bottom:20px;
       transition: transform 0.3s, box-shadow 0.3s;}
.card:hover {transform: scale(1.05); box-shadow:4px 4px 25px rgba(0,0,0,0.25);}
</style>
""", unsafe_allow_html=True)

# -------------------
# State init
# -------------------
if "diet_log" not in st.session_state:
    st.session_state.diet_log = pd.DataFrame(columns=["Timestamp","FileKey","FileName","Food Name","Calories"])
if "diet_keys" not in st.session_state:
    st.session_state.diet_keys = set()

# -------------------
# Sidebar navigation
# -------------------
st.sidebar.markdown("## 🍴 Navigation")
nav = st.sidebar.radio("Go to:", ["About", "Excess Calories Effect", "Upload & Predict", "Diet Tracking Table", "Quiz"])

# -------------------
# About
# -------------------
# -------------------
# -------------------
## -------------------
# 🌈 About Page — Animated Gradient + Typing Reveal + Changing Card Colors
# -------------------
if nav == "About":
    import streamlit as st
    import time

    st.markdown("""
    <style>
    /* 🌈 Animated background */
    body {
        background: linear-gradient(-45deg, #fff3e0, #ffe0b2, #ffccbc, #f8bbd0, #f1f8e9);
        background-size: 500% 500%;
        animation: gradientMove 14s ease infinite;
        font-family: 'Poppins', sans-serif;
    }

    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    h1.animated-title {
        text-align: center;
        color: #FF5733;
        background-color: rgba(255, 245, 229, 0.9);
        padding: 15px;
        border-radius: 15px;
        animation: fadeInDown 1.2s ease;
    }

    @keyframes fadeInDown {
        0% {opacity: 0; transform: translateY(-20px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    .blink-food {
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% {opacity: 1;}
        50% {opacity: 0.4;}
        100% {opacity: 1;}
    }

    /* ✨ Animated feature card with shifting pastel colors */
    .statement-card {
        background: linear-gradient(135deg, #ffe0b2, #f8bbd0, #f1f8e9, #fff9c4);
        background-size: 400% 400%;
        animation: colorShift 8s ease infinite;
        border-radius: 20px;
        padding: 15px;
        margin: 18px auto;
        width: 85%;
        box-shadow: 3px 3px 14px rgba(0,0,0,0.1);
    }

    @keyframes colorShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 class='animated-title'>🍽️ About Food Pro App</h1>", unsafe_allow_html=True)

    # Animated icons
    st.markdown("""
    <div style="text-align:center; margin:25px 0;">
        <img class="blink-food" src="https://img.icons8.com/color/96/apple.png" width="60" style="margin:0 10px;">
        <img class="blink-food" src="https://img.icons8.com/color/96/salad.png" width="60" style="margin:0 10px;">
        <img class="blink-food" src="https://img.icons8.com/color/96/broccoli.png" width="60" style="margin:0 10px;">
        <img class="blink-food" src="https://img.icons8.com/color/96/hamburger.png" width="60" style="margin:0 10px;">
    </div>
    """, unsafe_allow_html=True)

    # About items
    about_items = [
        ("🍽️ 🤖 Advanced Food Recognition App", "AI-based recognition of foods from images."),
        ("🍴 📸 Predicts food from uploaded images", "Upload any food photo and get instant prediction."),
        ("🧂 🥗 Shows ingredients, calories & substitutes", "See detailed ingredients and healthy substitutes."),
        ("🔍 📌 Suggests similar recipes", "Get suggestions for similar dishes to try."),
        ("💪 🏃‍♂️ Wellness tips & exercises", "Tips on staying healthy and managing calories."),
        ("🎯 🏆 Gamification & emoji feedback", "Engage with fun feedback and rewards for usage."),
        ("🍎 🔬 AI calorie estimation included!", "Estimate calories even if the database info is missing.")
    ]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#FF5733;'>✨ Key Highlights</h3>", unsafe_allow_html=True)

    # Typing animation for each card
    for title, info in about_items:
        card_placeholder = st.empty()

        # Type title
        for i in range(len(title) + 1):
            card_placeholder.markdown(
                f"<div class='statement-card'><h3 style='text-align:center;'>{title[:i]}</h3></div>",
                unsafe_allow_html=True
            )
            time.sleep(0.03)

        # Type info
        info_placeholder = st.empty()
        for j in range(len(info) + 1):
            info_placeholder.markdown(
                f"<div class='statement-card'><p style='text-align:center;font-style:italic;'>{info[:j]}</p></div>",
                unsafe_allow_html=True
            )
            time.sleep(0.02)

        time.sleep(0.4)

    st.success("🎉 Welcome to the smartest AI-powered Food & Wellness Assistant!")

# -------------------
# -------------------
elif nav == "Excess Calories Effect":
    import streamlit as st
    import time

    # Initialize session state to track animation
    if 'animated' not in st.session_state:
        st.session_state.animated = False

    # -------------------
    # CSS for background & statements
    # -------------------
    st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #fff3e0, #ffe0b2, #f8bbd0, #f1f8e9, #fff9c4);
        background-size: 500% 500%;
        animation: gradientShift 14s ease infinite;
        font-family: 'Poppins', sans-serif;
    }
    @keyframes gradientShift {0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;}}

    h1.excess-title {
        text-align:center; color:#FF4D4D;
        background-color: rgba(255,255,255,0.8);
        padding: 12px; border-radius: 12px;
        display: inline-block; margin-bottom: 20px;
        animation: popIn 1.5s ease;
    }
    @keyframes popIn {0% {opacity:0; transform:scale(0.8);} 100% {opacity:1; transform:scale(1);}}

    .statement-line {
        border-radius:12px; padding:12px 18px; margin:10px auto; width:80%;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1); font-size:17px; text-align:left; color:#333;
        background-color: rgba(255,255,255,0.7);
    }
    .bg1 {background-color:#FFF5E5;} .bg2 {background-color:#E8F5E9;}
    .bg3 {background-color:#E3F2FD;} .bg4 {background-color:#FFF3E0;} .bg5 {background-color:#F3E5F5;}
    </style>
    """, unsafe_allow_html=True)

    # 🌟 Title
    st.markdown("<h1 class='excess-title'>🔥 Excess Calories Effect</h1>", unsafe_allow_html=True)

    # -------------------
    # Statements with animation (one by one)
    # -------------------
    statements = [
        "⚡ Weight Gain → Stored as body fat",
        "❤️ Heart Issues → High cholesterol & BP",
        "😴 Fatigue → Low energy & sluggishness",
        "🤯 Mental Fog → Difficulty concentrating",
        "🍩 Diseases → Diabetes, fatty liver, obesity"
    ]
    bg_classes = ["bg1", "bg2", "bg3", "bg4", "bg5"]

    placeholders = [st.empty() for _ in statements]

    if not st.session_state.animated:
        for i, statement in enumerate(statements):
            bg_class = bg_classes[i % len(bg_classes)]
            text = ""
            for char in statement:
                text += char
                placeholders[i].markdown(f"<div class='statement-line {bg_class}'>{text}</div>", unsafe_allow_html=True)
                time.sleep(0.03)
            time.sleep(0.2)
        st.session_state.animated = True
    else:
        for i, statement in enumerate(statements):
            bg_class = bg_classes[i % len(bg_classes)]
            placeholders[i].markdown(f"<div class='statement-line {bg_class}'>{statement}</div>", unsafe_allow_html=True)

    st.success("💪 Keep your calorie intake balanced — your body will thank you!")

    # -------------------
    # Quick Features Section
    # -------------------
    st.markdown("### ⚡ Quick Features")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🧘 Reduce Calories"):
            st.info("""
            ✅ Sun Salutation 🌞🧘  
            ✅ Cardio 🏃‍♂️ 30 mins/day  
            ✅ Walk 🚶‍♀️ 5000+ steps  
            ✅ Hydrate 💧  
            ✅ Stretching 🤸‍♂️
            """)

    with col2:
        if st.button("🚫 Foods to Avoid"):
            st.warning("""
            ❌ Sugary Drinks 🥤  
            ❌ Fast Food 🍔🍟  
            ❌ Fried Snacks 🍟  
            ❌ Excess Sweets 🍰🍫  
            ❌ Processed Foods 🥫
            """)

    with col3:
        emoji_feedback = st.radio("💬 Rate this app:", ["😋 Loved it", "😐 Okay", "😞 Dislike"])

    with col4:
        if st.button("🍽️ SafeTips"):
            st.markdown("""
            <div style="background-color:#fff3e6; padding:15px; border-radius:10px;
            box-shadow:2px 2px 10px rgba(0,0,0,0.2);">
                <h4>🥦 Smart Eating Tips</h4>
                <p>🍽️ Eat smaller portions on a plate</p>
                <p>💧 Drink water before meals</p>
                <p>🥗 Add veggies & salads first</p>
                <p>⏱️ Eat slowly, enjoy every bite</p>
                <p>🛑 Stop when you feel 80% full</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------
if nav == "Upload & Predict":
    st.markdown('<div class="header">🍽️ Upload & Predict</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("📸 Upload Food Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    use_ai = st.sidebar.checkbox("🔬 Enable AI Calories Estimation", value=True)

    if uploaded_files:
        cols = st.columns(len(uploaded_files))
        total_calories = 0  # initialize here

        for i, uploaded_file in enumerate(uploaded_files):
            temp_path = f"temp_{i}.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            food = predict_food(temp_path)
            ingredients, calories, substitute = get_recipe(food)

            if use_ai:
                cal_est = estimate_calories_from_image(temp_path, food)
                if not calories or str(calories).lower() == "nan":
                    calories = cal_est if cal_est else "Not found"

            sim = similar_recipes(food)

            file_key = f"{uploaded_file.name}|{uploaded_file.size}"
            if file_key not in st.session_state.diet_keys:
                st.session_state.diet_log.loc[len(st.session_state.diet_log)] = [
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    file_key, uploaded_file.name, food, calories
                ]
                st.session_state.diet_keys.add(file_key)

            with cols[i]:
                st.image(temp_path, caption=f"Image {i+1}", use_container_width=True)
                st.markdown(f"""
                <div class='card'>
                    <h3>🍽️ {food}</h3>
                    <p>🧂 <b>Ingredients:</b> {ingredients}</p>
                    <p>🔥 <b>Calories:</b> <span style='color:red'>{calories} kcal</span></p>
                    <p>🥗 <b>Healthy Substitute:</b> {substitute}</p>
                    <p>🔍 <b>Similar Recipes:</b> {', '.join(sim)}</p>
                </div>
                """, unsafe_allow_html=True)

                try:
                    total_calories += float(calories)
                except:
                    pass

        # Display total calories
        st.markdown(f"### 🧾 Total Calories: *{round(total_calories, 2)} kcal*")

        # Function to animate multiple messages line by line
        def animated_messages(messages, message_type="info", delay=0.4):
            for msg in messages:
                if message_type == "warning":
                    st.warning(msg)
                elif message_type == "info":
                    st.info(msg)
                elif message_type == "success":
                    st.success(msg)
                else:
                    st.write(msg)
                time.sleep(delay)

        # Animated wellness messages based on total calories
        if total_calories > 500:
            tips = [
                "⚠️ High intake! Do 45 min cardio 🏃‍♂️💪",
                "💧 Drink plenty of water to stay hydrated",
                "🥗 Add fiber-rich vegetables or a side salad",
                "😴 Avoid heavy late-night meals",
                "🧘‍♂️ Try 10 min of stretching or yoga post-meal"
            ]
            animated_messages(tips, message_type="warning")

        elif total_calories > 300:
            tips = [
                "⚡ Moderate intake. Take a 30 min walk 🚶‍♀️",
                "💧 Stay hydrated",
                "🥗 Include a light salad or fruits"
            ]
            animated_messages(tips, message_type="info")

        else:
            tips = [
                "🌿 Light meal. Great balance!",
                "💧 Normal water intake",
                "😊 Perfect for maintaining energy without extra effort"
            ]
            animated_messages(tips, message_type="success")

# Diet Tracking Table
# -------------------
if nav == "Diet Tracking Table":
    st.markdown('<div class="header">📋 Diet Tracking Log</div>', unsafe_allow_html=True)

    if st.button("🧹 Clear Log"):
        st.session_state.diet_log = pd.DataFrame(
            columns=["Timestamp","FileKey","FileName","Food Name","Calories"]
        )
        st.session_state.diet_keys = set()
        st.success("Diet log cleared.")

    if st.session_state.diet_log.empty:
        st.info("No entries yet. Upload images to log your food.")
    else:
        df_display = st.session_state.diet_log.drop(columns=["FileKey"], errors="ignore")
        st.dataframe(
            df_display.sort_values(by="Timestamp", ascending=False)
            .reset_index(drop=True)
        )
        # Clean and convert Calories column safely
        df_display["Calories"] = pd.to_numeric(df_display["Calories"], errors='coerce')
        total_logged = df_display["Calories"].dropna().sum()
        st.markdown(f"**Total logged calories:** {round(total_logged, 2)} kcal")
# -------------------
# Quiz Page with Right-Side Feedback
# -------------------
if nav == "Quiz":
    st.markdown('<div class="header">📝 Food Quiz</div>', unsafe_allow_html=True)

    # Fade in/out animation for subtitle
    st.markdown("""
    <style>
    @keyframes fadeInOut {
        0% {opacity: 0;}
        50% {opacity: 1;}
        100% {opacity: 0;}
    }
    .fade-text {
        font-size: 18px;
        color: #333;
        font-weight: bold;
        animation: fadeInOut 4s ease-in-out infinite;
        text-align: center;
        margin-bottom: 20px;
    }
    .feedback {
        font-size: 18px;
        font-weight: bold;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fade-text">Test your knowledge about food, calories, and nutrition!</div>', unsafe_allow_html=True)

    # Custom CSS for colored questions
    st.markdown("""
    <style>
    .q1 {color: #FF5733; font-weight:bold; animation: popIn 1s;}
    .q2 {color: #33C1FF; font-weight:bold; animation: popIn 1s;}
    .q3 {color: #75FF33; font-weight:bold; animation: popIn 1s;}
    .q4 {color: #FF33A6; font-weight:bold; animation: popIn 1s;}
    .q5 {color: #FFA533; font-weight:bold; animation: popIn 1s;}
    @keyframes popIn {
        0% {transform: scale(0.8); opacity:0;}
        100% {transform: scale(1); opacity:1;}
    }
    </style>
    """, unsafe_allow_html=True)

    score = 0

    # Question 1
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="q1">1️⃣ Which food is highest in protein?</p>', unsafe_allow_html=True)
        q1 = st.radio("Select an answer:", ["Burger", "Egg", "Cake"], key="q1")
    with col2:
        if q1:
            if q1 == "Egg":
                st.markdown('<p class="feedback" style="color:green;">✅ Correct</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="feedback" style="color:red;">❌</p>', unsafe_allow_html=True)
    if q1 == "Egg":
        score += 1

    # Question 2
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="q2">2️⃣ Which has the lowest calories per 100g?</p>', unsafe_allow_html=True)
        q2 = st.radio("Select an answer:", ["Cheese", "Apple", "Chocolate"], key="q2")
    with col2:
        if q2:
            if q2 == "Apple":
                st.markdown('<p class="feedback" style="color:green;">✅ Correct</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="feedback" style="color:red;">❌</p>', unsafe_allow_html=True)
    if q2 == "Apple":
        score += 1

    # Question 3
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="q3">3️⃣ Which is a healthy fat source?</p>', unsafe_allow_html=True)
        q3 = st.radio("Select an answer:", ["Butter", "Avocado", "Bacon"], key="q3")
    with col2:
        if q3:
            if q3 == "Avocado":
                st.markdown('<p class="feedback" style="color:green;">✅ Correct</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="feedback" style="color:red;">❌</p>', unsafe_allow_html=True)
    if q3 == "Avocado":
        score += 1

    # Question 4
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="q4">4️⃣ What should you drink before meals to reduce calorie intake?</p>', unsafe_allow_html=True)
        q4 = st.radio("Select an answer:", ["Soda", "Water", "Juice"], key="q4")
    with col2:
        if q4:
            if q4 == "Water":
                st.markdown('<p class="feedback" style="color:green;">✅ Correct</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="feedback" style="color:red;">❌</p>', unsafe_allow_html=True)
    if q4 == "Water":
        score += 1

    # Question 5
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="q5">5️⃣ Which is a high-calorie dessert?</p>', unsafe_allow_html=True)
        q5 = st.radio("Select an answer:", ["Donuts", "Fruit Salad", "Yogurt"], key="q5")
    with col2:
        if q5:
            if q5 == "Donuts":
                st.markdown('<p class="feedback" style="color:green;">✅ Correct</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="feedback" style="color:red;">❌</p>', unsafe_allow_html=True)
    if q5 == "Donuts":
        score += 1

    # Submit button
    if st.button("Submit Quiz"):
        st.success(f"🎉 You scored {score}/5")
        if score == 5:
            st.balloons()
        elif score >= 3:
            st.info("👍 Good job! Keep learning.")
        else:
            st.warning("⚠️ Keep practicing and improve your food knowledge.")

    # -------------------
    # Line Graph Visualization
    # -------------------
    import pandas as pd
    import altair as alt
    import numpy as np

    # Prepare dataframe
    user_data = {
        "Question": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "Result": [
            1 if q1 == "Egg" else 0,
            1 if q2 == "Apple" else 0,
            1 if q3 == "Avocado" else 0,
            1 if q4 == "Water" else 0,
            1 if q5 == "Donuts" else 0,
        ]
    }
    df = pd.DataFrame(user_data)

    # Interpolate for smooth animation
    smooth_x = []
    smooth_y = []
    for i, row in df.iterrows():
        smooth_x.extend([row["Question"]] * 10)
        smooth_y.extend(np.linspace(0, row["Result"] * 100, 10))

    df_smooth = pd.DataFrame({"Question": smooth_x, "Percentage": smooth_y})

    # Line chart
    line_chart = alt.Chart(df_smooth).mark_line(point=True).encode(
        x=alt.X('Question', sort=None, title='Question'),
        y=alt.Y('Percentage', title='Correctness (%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.value('dodgerblue')
    ).properties(
        width=600,
        height=400,
        title="📈 Quiz Result Line Graph"
    )

    st.altair_chart(line_chart)
