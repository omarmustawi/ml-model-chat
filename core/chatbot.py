import re
import pandas as pd

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



from sentence_transformers import SentenceTransformer

from core.config import DATA_PATH

from utils.text_utils import normalize_text


from core.helpers import (
    balance_intents,
    build_intent_dataset,
    is_greeting,
    find_meal_name_in_text,
    extract_goal,
    extract_meal_type,
    extract_workout_intent,
    find_meal_name_in_text,
    handle_greeting,
)


from core.recommendation import recommend_meals


from core.feedback_bandit import FeedbackBandit
from core.memory import ChatMemory, ask_clarification


from core.analysis import (
    analyze_meal,
    compare_meals,
    substitute_meal,
)


from core.planner import daily_plan, weekly_plan

# تحميل مودل embeddings (خفيف ويدعم عربي + إنجليزي)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


df = pd.read_csv(DATA_PATH)

df = df.drop(["Cuisine"], axis=1).copy()

# Strip spaces from string columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# Ensure numeric columns are numeric
numeric_cols = ["Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop duplicates if any
df = df.drop_duplicates().reset_index(drop=True)


df["goal_l"] = df["Goal"].apply(normalize_text)
df["meal_type_l"] = df["Meal Type"].apply(normalize_text)
df["name_l"] = df["Name"].apply(normalize_text)
df["ingredients_l"] = df["Ingredients"].apply(normalize_text)


df["meal_profile"] = (
    "name: "
    + df["Name"].astype(str)
    + " | goal: "
    + df["Goal"].astype(str)
    + " | meal type: "
    + df["Meal Type"].astype(str)
    +
    # " | cuisine: " + df["Cuisine"].astype(str) +
    " | ingredients: "
    + df["Ingredients"].astype(str)
)


# Normalized nutrition columns for ranking
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)


intent_df = build_intent_dataset(df)

intent_df = balance_intents(intent_df)

# intent_df = downsample(intent_df)


intent_texts = intent_df["text"].tolist()
intent_labels = intent_df["intent"].tolist()

intent_embeddings = embedder.encode(intent_texts, convert_to_tensor=False)

profile_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
profile_matrix = profile_vectorizer.fit_transform(df["meal_profile"])


bandit = FeedbackBandit(epsilon=0.15)

memory = ChatMemory()



def predict_intent_semantic(user_input, threshold=0.55):
    emb = embedder.encode([user_input])
    sims = cosine_similarity(emb, intent_embeddings)[0]

    idx = np.argmax(sims)
    score = sims[idx]

    if score < threshold:
        return "unknown", score

    return intent_labels[idx], score


def chatbot(user_text):

    # 1. Intent
    intent, score = predict_intent_semantic(user_text)
    # print(f"Predicted intent: {intent} (score: {score:.2f})")
    # intent = intent_model.predict([user_text])[0]

    # 2. 🤝 Greeting
    greeting_msg = None
    if is_greeting(text=user_text):
        if intent != "unknown":
            greeting_msg = handle_greeting(intent)["greeting"]
        else:
            return handle_greeting()

    # 3. Extract slots
    goal = extract_goal(user_text) or memory.last_goal
    meal_type = extract_meal_type(user_text) or memory.last_meal_type
    workout = extract_workout_intent(user_text)
    meal_name = find_meal_name_in_text(user_text) or memory.last_meal_name

    pending_intent = memory.pending_intent
    pending_options = memory.pending_options
    last_compare = memory.last_compare_meals

    # 🔥 HANDLE PENDING CLARIFICATION
    if pending_intent == "compare_meals" and pending_options:
        selected = find_meal_name_in_text(user_text)

        # إذا المستخدم اختار وجبة من الخيارات
        if isinstance(selected, str) and selected in pending_options:
            # إذا كان عندنا وجبة واحدة محفوظة سابقاً
            if last_compare and len(last_compare) == 1:
                meal_a = last_compare[0]
                meal_b = selected
            else:
                meal_a = pending_options[0]
                meal_b = selected

            # تنظيف الذاكرة
            memory.update(
                pending_intent=None,
                pending_options=None,
                compare_meals=[meal_a, meal_b],
            )

            result = compare_meals(meal_a, meal_b, goal=goal)

            return {"intent": "compare_meals", "comparison": result}

    # 4. Routing
    if intent == "recommend_meal":
        if not goal or not meal_type:
            return ask_clarification(goal=goal, meal_type=meal_type)

        top = recommend_meals(user_text, top_n=1, goal=goal, meal_type=meal_type)
        memory.update(intent, goal, meal_type)
        response = {
            "intent": intent,
            "goal": goal,
            "meal_type": meal_type,
            "results": top,
        }
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "list_meals":
        # list meals by inferred meal type or goal
        query = user_text
        top = recommend_meals(query, top_n=10, goal=goal, meal_type=meal_type)
        response = {
            "intent": intent,
            "goal": goal,
            "meal_type": meal_type,
            "results": top,
        }
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "pre_workout":
        top = recommend_meals(
            user_text,
            top_n=5,
            workout="pre_workout" if False else None,
            meal_type=meal_type,
        )
        # use nutrition scorer through query text
        response = {"intent": intent, "results": top}
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "post_workout":
        top = recommend_meals(user_text, top_n=5, meal_type=meal_type)
        response = {"intent": intent, "results": top}
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "daily_plan":
        g = goal or "maintenance"
        response = {"intent": intent, "goal": g, "plan": daily_plan(g)}
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "weekly_plan":
        g = goal or "maintenance"
        response = {"intent": intent, "goal": g, "plan": weekly_plan(g)}
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "analyze_meal":
        if meal_name is None:
            response = {"intent": intent, "error": "Please mention a meal name."}
            if greeting_msg:
                response["greeting"] = greeting_msg
            return response

        # row = df[df["Name"].str.lower() == meal_name].iloc[0]
        memory.update(intent, meal_name=meal_name)

        response = {
            "intent": intent,
            "analysis": analyze_meal(meal_name, style="colloquial"),
        }
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "substitute_meal":
        if meal_name is None:
            response = {"intent": intent, "error": "Please mention a meal name."}
            if greeting_msg:
                response["greeting"] = greeting_msg
            return response

        if isinstance(meal_name, list):
            response = {
                "intent": intent,
                "error": "I found multiple matching meals. Please specify one.",
                "candidates": meal_name[:5],
            }
            if greeting_msg:
                response["greeting"] = greeting_msg
            return response

        response = {
            "intent": intent,
            "substitutes": substitute_meal(meal_name, top_n=5, style="colloquial"),
        }
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    if intent == "compare_meals":
        names = []

        for m in df["Name"].tolist():
            if normalize_text(m) in normalize_text(user_text):
                if m not in names:
                    names.append(m)
        # fallback split
        parts = re.split(r"\band\b", user_text, flags=re.IGNORECASE)
        for p in parts:
            m = find_meal_name_in_text(p)
            if m and m not in names:
                names.append(m)

        if len(names) < 2:
            response = {
                "intent": intent,
                "error": "Please mention 2 meal names clearly.",
            }
            if greeting_msg:
                response["greeting"] = greeting_msg
            return response

        # 🔴 clarification step
        if isinstance(names[0], list) or isinstance(names[1], list):

            options = []
            matched_compare_meals = [] = []

            if isinstance(names[0], list):
                options.extend(names[0][:5])
            if isinstance(names[1], list):
                options.extend(names[1][:5])

            if isinstance(names[0], str):
                matched_compare_meals.append(names[0])
            if isinstance(names[1], str):
                matched_compare_meals.append(names[1])

            response = {
                "intent": "clarification",
                "message": "I couldn't clearly identify two meals.",
                "question": "Which meals do you want to compare?",
                "options": options,
            }

            memory.update(
                intent="compare_meals",
                compare_meals=matched_compare_meals,
                pending_intent="compare_meals",
                pending_options=options,
            )

            if greeting_msg:
                response["greeting"] = greeting_msg

            return response

        result = compare_meals(names[0], names[1], goal=goal)
        memory.update(
            intent="compare_meals",
            compare_meals=[names[0], names[1]],
            meal_name=names[1],
        )

        if "error" in result:
            return {"intent": intent, "message": result["error"]}

        return {"intent": intent, "comparison": result}

    if intent == "unknown":
        response = {
            "intent": "unknown",
            "message": "I didn't quite understand you 😅 Could you rephrase it?",
        }
        if greeting_msg:
            response["greeting"] = greeting_msg
        return response

    # return "OK 👍"
