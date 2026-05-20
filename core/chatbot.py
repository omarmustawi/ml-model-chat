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

from typing import Any, Optional
import joblib
import os

intent_model = joblib.load(os.path.join("models", "intent_model.joblib"))


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


bandit = FeedbackBandit(epsilon=0.1)

memory = ChatMemory()


def reset_pending():
    memory.update(
        pending_intent=None,
        pending_options=None,
    )


def reset_context():
    memory.update(
        last_goal=None,
        last_meal_type=None,
        last_meal_name=None,
        last_compare_meals=None,
        pending_intent=None,
        pending_options=None,
    )


def is_topic_changed(current_intent, user_text):

    last_intent = memory.pending_intent

    if not last_intent:
        return False

    # إذا النية مختلفة والرسالة طويلة
    if current_intent != last_intent and len(user_text.split()) > 4:
        return True

    return False


def _attach_greeting(
    response: dict[str, Any], greeting_msg: Optional[str]
) -> dict[str, Any]:
    if greeting_msg:
        response["greeting"] = greeting_msg
    return response


def _save_and_return(
    response: dict[str, Any],
    goal: Optional[str] = None,
    meal_type: Optional[str] = None,
    meal_name: Optional[str] = None,
    compare_meals: Optional[list[str]] = None,
    pending_intent: Optional[str] = None,
    pending_options: Optional[list[str]] = None,
) -> dict[str, Any]:

    memory.update(
        last_goal=goal,
        last_meal_type=meal_type,
        last_meal_name=meal_name,
        last_compare_meals=compare_meals,
        pending_intent=pending_intent,
        pending_options=pending_options,
    )

    # memory.add_message("assistant", response)
    return response


def _normalize_meal_name(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return value[0]
    return None


def predict_intent_semantic(user_input, threshold=0.55):
    emb = embedder.encode([user_input])
    sims = cosine_similarity(emb, intent_embeddings)[0]

    idx = np.argmax(sims)
    score = sims[idx]

    if score < threshold:
        return "unknown", score

    return intent_labels[idx], score


def chatbot(user_text: str) -> dict[str, Any]:
    user_text = (user_text or "").strip()

    if not user_text:
        return {
            "intent": "unknown",
            "results": "Please type a message.",
        }

    # memory.add_message("user", user_text)

    # 1. Intent
    if memory.pending_intent:
        intent = memory.pending_intent
        # score = 1.0
    else:
        intent_semantic, score = predict_intent_semantic(user_text)
        intent = intent_model.predict([user_text])[0]
        probs = intent_model.predict_proba([user_text])[0]
        max_prob = max(probs)

        print(
            f"intent: {intent}, max_prob: {max_prob},  intent_semantic: {intent_semantic}, score: {score}"
        )
        if score > max_prob:
            intent = intent_semantic

    # 2. 🤝 Greeting
    greeting_msg = None
    if is_greeting(text=user_text):
        if intent == "unknown":
            response = handle_greeting()
            # memory.add_message("assistant", response)
            return response

        greeting_data = handle_greeting(intent)

        if isinstance(greeting_data, dict):
            greeting_msg = greeting_data.get("greeting")
        else:
            greeting_msg = greeting_data

    # 3. Extract slots
    goal = extract_goal(user_text) or memory.last_goal
    meal_type = extract_meal_type(user_text) or memory.last_meal_type
    workout = extract_workout_intent(user_text)
    meal_name = find_meal_name_in_text(user_text)
    if isinstance(meal_name, list) and len(meal_name) == 1:
        meal_name = meal_name[0]

    pending_intent = memory.pending_intent
    pending_options = memory.pending_options or []
    last_compare = memory.last_compare_meals or []

    if is_topic_changed(intent, user_text):
        reset_context()

    print(
        f"Meal name: {meal_name}- Goal: {goal} - Meal type: {meal_type} - Workout: {workout}"
    )

    if intent == "recommend_meal":
        if not goal or not meal_type:
            response = ask_clarification(intent=intent, goal=goal, meal_type=meal_type)
            _attach_greeting(response, greeting_msg)
            return _save_and_return(
                response,
                goal=goal,
                meal_type=meal_type,
                pending_intent="recommend_meal",
            )

        top = recommend_meals(user_text, top_n=1, goal=goal, meal_type=meal_type)

        response = {
            "intent": intent,
            "goal": goal,
            "meal_type": meal_type,
            "results": top,
        }

        _attach_greeting(response, greeting_msg)

        return _save_and_return(
            response,
            goal=goal,
            meal_type=meal_type,
        )

    if intent == "list_meals":
        if not goal or not meal_type:
            response = ask_clarification(intent=intent, goal=goal, meal_type=meal_type)
            _attach_greeting(response, greeting_msg)
            return _save_and_return(
                response,
                goal=goal,
                meal_type=meal_type,
                pending_intent="list_meals",
            )

        top = recommend_meals(user_text, top_n=10, goal=goal, meal_type=meal_type)

        response = {
            "intent": intent,
            "goal": goal,
            "meal_type": meal_type,
            "results": top,
        }

        _attach_greeting(response, greeting_msg)
        return _save_and_return(response, goal=goal, meal_type=meal_type)

    if intent == "pre_workout":
        top = recommend_meals(
            user_text,
            top_n=5,
            workout=workout or "pre_workout",
            meal_type=meal_type,
        )

        response = {"intent": intent, "results": top}
        _attach_greeting(response, greeting_msg)
        return _save_and_return(
            response,
            goal=goal,
            meal_type=meal_type,
            workout=workout,
        )

    if intent == "post_workout":
        top = recommend_meals(user_text, top_n=5, meal_type=meal_type)

        response = {"intent": intent, "results": top}
        _attach_greeting(response, greeting_msg)
        return _save_and_return(
            response,
            goal=goal,
            meal_type=meal_type,
            workout=workout,
        )

    if intent == "daily_plan":
        if goal is None:
            response = ask_clarification(intent=intent, goal=goal)
            _attach_greeting(response, greeting_msg)
            return _save_and_return(response, goal=goal, pending_intent="daily_plan")

        # g = goal or "maintenance"

        response = {"intent": intent, "goal": goal, "plan": daily_plan(goal)}
        _attach_greeting(response, greeting_msg)
        return _save_and_return(
            response,
            goal=goal,
            meal_type=meal_type,
        )

    if intent == "weekly_plan":
        if goal is None:
            response = ask_clarification(intent=intent, goal=goal)
            _attach_greeting(response, greeting_msg)
            return _save_and_return(response, goal=goal, pending_intent="weekly_plan")

        # g = goal or "maintenance"
        response = {"intent": intent, "goal": goal, "plan": weekly_plan(goal)}
        _attach_greeting(response, greeting_msg)
        return _save_and_return(
            response,
            goal=goal,
            meal_type=meal_type,
        )

    if intent == "analyze_meal":
        if memory.pending_intent == "analyze_meal" and memory.pending_options:
            selected = _normalize_meal_name(find_meal_name_in_text(user_text))
            print(
                f"Pending analyze_meal - user selected: {selected} from options: {memory.pending_options}"
            )
            if selected and selected in memory.pending_options:
                meal_name = selected
                memory.update(
                    pending_intent=None,
                    pending_options=None,
                )

        if not meal_name:
            response = ask_clarification(
                intent=intent,
                meal_name=meal_name,
                options=[],
            )

            _attach_greeting(response, greeting_msg)

            return _save_and_return(
                response, meal_name=meal_name, pending_intent="analyze_meal"
            )

        if isinstance(meal_name, list):

            candidates = meal_name[:5]
            response = ask_clarification(
                intent=intent,
                meal_name=None,
                options=candidates,
            )
            _attach_greeting(response, greeting_msg)
            return _save_and_return(
                response,
                meal_name=meal_name,
                pending_intent="analyze_meal",
                pending_options=candidates,
            )

        response = {
            "intent": intent,
            "results": analyze_meal(meal_name, style="colloquial"),
        }
        _attach_greeting(response, greeting_msg)

        return _save_and_return(
            response,
            meal_name=meal_name,
            pending_intent=None,
            pending_options=None,
        )

    if intent == "substitute_meal":
        if meal_name is None:
            response = ask_clarification(
                intent=intent,
                meal_name=meal_name,
                options=[],
            )

            _attach_greeting(response, greeting_msg)
            return _save_and_return(
                response,
                meal_name=meal_name,
                pending_intent="substitute_meal",
            )

        if isinstance(meal_name, list):

            candidates = meal_name[:5]
            response = ask_clarification(
                intent=intent,
                meal_name=None,
                options=candidates,
            )
            _attach_greeting(response, greeting_msg)
            return _save_and_return(
                response,
                meal_name=meal_name,
                pending_intent="substitute_meal",
                pending_options=candidates,
            )

        response = {
            "intent": intent,
            "results": substitute_meal(meal_name, top_n=5, style="colloquial"),
        }
        _attach_greeting(response, greeting_msg)
        return _save_and_return(
            response,
            meal_name=meal_name,
            pending_intent=None,
            pending_options=None,
        )

    if intent == "compare_meals":
        # =========================================================
        # 1. إذا كان المستخدم يرد على clarification سابق
        # =========================================================
        if memory.pending_intent == "compare_meals":

            selected = _normalize_meal_name(find_meal_name_in_text(user_text))
            print(f"Pending compare selection: {selected}")

            previous_meals = memory.last_compare_meals or []

            if selected and previous_meals:

                meal_a = previous_meals[0]
                meal_b = selected

                if normalize_text(meal_a) == normalize_text(meal_b):

                    response = {
                        "intent": intent,
                        "results": "Please choose a different meal for comparison.",
                    }

                    _attach_greeting(response, greeting_msg)

                    return _save_and_return(
                        response,
                        compare_meals=[meal_a],
                        goal=goal,
                        # pending_intent="compare_meals",
                        # pending_options=memory.pending_options,
                        pending_intent=None,
                        pending_options=None,
                    )

                # تنظيف الـ pending state
                memory.update(
                    pending_intent=None,
                    pending_options=None,
                )

                result = compare_meals(
                    meal_a,
                    meal_b,
                    goal=goal,
                )

                response = {
                    "intent": "compare_meals",
                    "results": result,
                }

                _attach_greeting(response, greeting_msg)

                return _save_and_return(
                    response,
                    compare_meals=[meal_a, meal_b],
                    goal=goal,
                    pending_intent=None,
                    pending_options=None,
                )

        # =========================================================
        # 2. استخراج أسماء الوجبات من الرسالة
        # =========================================================
        found_names = []

        normalized_user_text = normalize_text(user_text)

        # exact match
        for meal in df["Name"].dropna().tolist():

            normalized_meal = normalize_text(meal)

            if normalized_meal in normalized_user_text and meal not in found_names:
                found_names.append(meal)

        print(f"Found compare meals after exact match: {found_names}")
        # fallback fuzzy extraction
        if len(found_names) < 2:

            parts = re.split(
                r"\band\b|,",
                user_text,
                flags=re.IGNORECASE,
            )
            for part in parts:

                match = find_meal_name_in_text(part)

                if isinstance(match, list):

                    for candidate in match[:5]:

                        if candidate not in found_names:
                            found_names.append(candidate)

                elif isinstance(match, str):

                    if match not in found_names:
                        found_names.append(match)

        print(f"Found compare meals: {found_names}")

        # =========================================================
        # 3. لا يوجد أي وجبة
        # =========================================================
        if len(found_names) == 0:

            response = ask_clarification(
                intent=intent,
                meal_name=None,
                options=[],
            )
            _attach_greeting(response, greeting_msg)
            return _save_and_return(
                response,
                meal_name=None,
                goal=goal,
                pending_intent="compare_meals",
            )

        # =========================================================
        # 4. يوجد وجبة واحدة فقط → clarification
        # =========================================================
        if len(found_names) == 1:

            first_meal = found_names[0]

            candidate_options = (
                df[df["Name"] != first_meal]["Name"]
                .dropna()
                .sample(min(5, len(df) - 1))
                .tolist()
            )

            response = ask_clarification(
                intent="compare_meals",
                compare_meals=[first_meal],
                options=candidate_options,
            )

            _attach_greeting(response, greeting_msg)

            return _save_and_return(
                response,
                compare_meals=[first_meal],
                goal=goal,
                pending_intent="compare_meals",
                pending_options=candidate_options,
            )

        # =========================================================
        # 5. يوجد وجبتين أو أكثر
        # =========================================================
        meal_a = found_names[0]
        meal_b = found_names[1]

        # منع مقارنة نفس الوجبة
        if normalize_text(meal_a) == normalize_text(meal_b):

            response = {
                "intent": intent,
                "results": "Please provide two different meals.",
            }

            _attach_greeting(response, greeting_msg)

            return _save_and_return(
                response,
                goal=goal,
            )

        # =========================================================
        # 6. تنفيذ المقارنة
        # =========================================================
        result = compare_meals(
            meal_a,
            meal_b,
            goal=goal,
        )

        response = {
            "intent": intent,
            "results": result,
        }

        _attach_greeting(response, greeting_msg)

        return _save_and_return(
            response,
            # compare_meals=[meal_a, meal_b],
            compare_meals=None,
            pending_intent=None,
            pending_options=None,
            goal=None,
        )

    if intent == "unknown":
        response = {
            "intent": "unknown",
            "results": "I didn't quite understand you 😅 Could you rephrase it?",
        }
        _attach_greeting(response, greeting_msg)
        return _save_and_return(response)
