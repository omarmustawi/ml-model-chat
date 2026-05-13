# # Cell 1 — Imports and configuration
# import os
# import re
# import json
# import math
# import random
# import joblib
# import difflib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from collections import defaultdict

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression, Ridge
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     mean_absolute_error,
#     r2_score,
# )
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.utils import resample

# from sentence_transformers import SentenceTransformer

# # تحميل مودل embeddings (خفيف ويدعم عربي + إنجليزي)
# embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# DATA_PATH = "arab_meals_120.csv"
# MODEL_DIR = "models"

# RANDOM_STATE = 42


# df = pd.read_csv(DATA_PATH)


# # Cell 2 - Greeting handler

# GREETINGS = [
#     "hello",
#     "hi",
#     "hey",
#     "مرحبا",
#     "السلام",
#     "السلام عليكم",
#     "هلا",
#     "صباح الخير",
#     "مساء الخير",
# ]


# def is_greeting(text):
#     text = text.lower()
#     words = re.findall(r"\b\w+\b", text)  # tokenize
#     return any(g in words for g in GREETINGS)


# def handle_greeting(intent=None):
#     if intent:
#         return {
#             "intent": intent,
#             "greeting": "Hello, ",
#         }
#     return {
#         "intent": "greeting",
#         "greeting": "👋 How can I help you? You can request a meal, meal analysis, or a diet plan.",
#     }


# df = df.drop(["Cuisine"], axis=1).copy()

# # Strip spaces from string columns
# for col in df.columns:
#     if df[col].dtype == "object":
#         df[col] = df[col].astype(str).str.strip()

# # Ensure numeric columns are numeric
# numeric_cols = ["Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# # Drop duplicates if any
# df = df.drop_duplicates().reset_index(drop=True)



# def normalize_text(x):
#     if isinstance(x, (list, tuple, np.ndarray)):
#         x = " ".join([str(i) for i in x if i is not None])

#     try:
#         if pd.isna(x):
#             return ""
#     except:
#         pass

#     return re.sub(r"\s+", " ", str(x).strip().lower())



# df["goal_l"] = df["Goal"].apply(normalize_text)
# df["meal_type_l"] = df["Meal Type"].apply(normalize_text)
# df["name_l"] = df["Name"].apply(normalize_text)
# df["ingredients_l"] = df["Ingredients"].apply(normalize_text)




# df["meal_profile"] = (
#     "name: "
#     + df["Name"].astype(str)
#     + " | goal: "
#     + df["Goal"].astype(str)
#     + " | meal type: "
#     + df["Meal Type"].astype(str)
#     +
#     # " | cuisine: " + df["Cuisine"].astype(str) +
#     " | ingredients: "
#     + df["Ingredients"].astype(str)
# )


# # Normalized nutrition columns for ranking
# for col in numeric_cols:
#     col_min = df[col].min()
#     col_max = df[col].max()
#     df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)



# # Cell 6 — Synthetic intent dataset
# id="fix1"
# from sklearn.utils import resample

# def balance_intents(df):
#     max_count = df['intent'].value_counts().max()
#     balanced = []

#     for intent in df['intent'].unique():
#         subset = df[df['intent'] == intent]

#         if len(subset) < max_count:
#             subset = resample(subset,
#                              replace=True,
#                              n_samples=max_count,
#                              random_state=42)

#         balanced.append(subset)

#     return pd.concat(balanced)



# id="fix2"
# def downsample(df, max_per_class=200):
#     parts = []
#     for intent in df['intent'].unique():
#         subset = df[df['intent'] == intent]
#         subset = subset.sample(min(len(subset), max_per_class), random_state=42)
#         parts.append(subset)
#     return pd.concat(parts)


# id="fix3"
# def expand_templates(templates, times=20):
#     new = []
#     for _ in range(times):
#         new.extend(templates)
#     return new



# GOAL_SYNONYMS = {
#     "muscle gain": [
#         "muscle gain",
#         "gain muscle",
#         "build muscle",
#         "bulking",
#         "increase muscle",
#         "تضخيم",
#         "بناء عضل",
#         "زيادة العضلات",
#     ],
#     "fat loss": [
#         "fat loss",
#         "lose fat",
#         "weight loss",
#         "cutting",
#         "lose weight",
#         "تنشيف",
#         "خسارة وزن",
#         "رجيم",
#         "حرق الدهون",
#     ],
#     "maintenance": [
#         "maintenance",
#         "maintain weight",
#         "stay fit",
#         "ثبات",
#         "المحافظة على الوزن",
#     ],
#     "endurance": [
#         "endurance",
#         "stamina",
#         "energy",
#         "athletic endurance",
#         "تحمل",
#         "طاقة",
#         "قدرة تحمل",
#     ],
# }

# MEAL_SYNONYMS = {
#     "breakfast": ["breakfast", "morning meal", "first meal", "فطور", "إفطار"],
#     "lunch": ["lunch", "midday meal", "noon meal", "غداء"],
#     "dinner": ["dinner", "evening meal", "night meal", "عشاء"],
#     "snack": ["snack", "light meal", "between meals", "سناك", "وجبة خفيفة"],
#     "dessert": ["dessert", "sweet", "after meal", "حلى", "تحلية"],
# }

# INTENT_LABELS = [
#     "recommend_meal",
#     "list_meals",
#     "daily_plan",
#     "weekly_plan",
#     "pre_workout",
#     "post_workout",
#     "analyze_meal",
#     "compare_meals",
#     "substitute_meal",
# ]


# def generate_goal_queries(goal):
#     synonyms = GOAL_SYNONYMS.get(goal.lower(), [goal])
#     meal_types = ["breakfast", "lunch", "dinner", "snack", "dessert"]
#     queries = []
#     for syn in synonyms:
#         queries.extend(
#             [
#                 f"recommend a meal for {syn}",
#                 f"give me a healthy meal for {syn}",
#                 f"I need food for {syn}",
#                 f"اقتراح وجبة لـ {syn}",
#                 f"اعطني وجبة مناسبة لـ {syn}",
#             ]
#         )
#         for mt in meal_types:
#             queries.extend(
#                 [
#                     f"recommend a {mt} for {syn}",
#                     f"give me a {mt} meal for {syn}",
#                     f"show me {mt} options for {syn}",
#                 ]
#             )
#     return queries


# def generate_mealtype_queries(meal_type):
#     synonyms = MEAL_SYNONYMS.get(meal_type.lower(), [meal_type])
#     queries = []
#     for syn in synonyms:
#         queries.extend(
#             [
#                 f"list {syn} meals",
#                 f"show me {syn} options",
#                 f"I want a {syn}",
#                 f"اعطني وجبة {syn}",
#                 f"أريد {syn} صحي",
#             ]
#         )
#     return queries


# def build_intent_dataset(df):
#     rows = []

#     # Goal-based and meal-type-based recommendation/listing examples
#     for _, row in df.iterrows():
#         goal = row["Goal"].lower()
#         meal_type = row["Meal Type"].lower()
#         meal_name = row["Name"]

#         # recommend_meal
#         for q in generate_goal_queries(goal)[:8]:
#             rows.append((q, "recommend_meal"))
#         for q in generate_mealtype_queries(meal_type)[:6]:
#             rows.append((q, "list_meals"))

#         # meal-specific intents
#         rows.extend(
#             [
#                 (f"analyze {meal_name}", "analyze_meal"),
#                 (f"what are the calories and macros of {meal_name}", "analyze_meal"),
#                 (f"compare {meal_name} with another meal", "compare_meals"),
#                 (f"suggest a substitute for {meal_name}", "substitute_meal"),
#             ]
#         )

#     # Generic plan / workout intents — multiple templates for each class
#     generic_templates = {
#         "daily_plan": [
#             "create a daily meal plan for fat loss",
#             "make me a one day meal plan for muscle gain",
#             "daily nutrition plan for maintenance",
#             "اعمل لي خطة يومية للتنشيف",
#             "plan my meals for today for endurance",
#             "give me a full day meal plan",
#             "build a daily nutrition plan for me",
#             "one day food plan for weight loss",
#         ],
#         "weekly_plan": [
#             "create a weekly meal plan for fat loss",
#             "make me a 7 day meal plan for muscle gain",
#             "weekly nutrition plan for maintenance",
#             "اعمل لي خطة أسبوعية للتضخيم",
#             "plan my meals for the week for endurance",
#             "give me a 7 day nutrition plan",
#             "build a weekly food plan",
#             "one week diet plan for fitness",
#         ],
#         "pre_workout": [
#             "recommend a pre workout meal",
#             "what should I eat before training",
#             "وجبة قبل التمرين",
#             "اقتراح سناك قبل التمرين",
#             "best meal before gym",
#             "what to eat before workout",
#             "need energy before training",
#             "suggest a light pre training meal",
#         ],
#         "post_workout": [
#             "recommend a post workout meal",
#             "what should I eat after training",
#             "وجبة بعد التمرين",
#             "اقتراح وجبة بعد التمرين",
#             "best meal after gym",
#             "what to eat after workout",
#             "need recovery food after training",
#             "suggest a high protein post training meal",
#         ],
#         "compare_meals": [
#             "compare two meals for me",
#             "which meal is better for fat loss",
#             "قارن بين وجبتين",
#             "أي وجبة أفضل للتضخيم",
#             "compare meal A and meal B",
#             "which option has better macros",
#             "help me compare foods",
#             "which is healthier between these meals",
#         ],
#         "analyze_meal": [
#             "analyze this meal",
#             "show me nutrition facts",
#             "ما هي القيم الغذائية لهذه الوجبة",
#             "احسب السعرات والماكروز",
#             "what are the calories in this meal",
#             "show protein carbs and fat",
#             "nutrition breakdown for this food",
#             "meal calorie analysis",
#         ],
#         "substitute_meal": [
#             "suggest a healthy substitution",
#             "replace this meal with a better option",
#             "اعطني بديل صحي",
#             "اقترح بديل لوجبة",
#             "find a similar healthier meal",
#             "recommend a substitute with similar taste",
#             "give me an alternative food",
#             "swap this meal with a lighter option",
#         ],
#         "recommend_meal": [
#             "recommend a meal",
#             "suggest me food",
#             "اعطني وجبة مناسبة",
#             "اقتراح وجبة صحية",
#             "give me something to eat",
#             "pick a meal for me",
#             "show me a good meal option",
#             "I want a meal suggestion",
#         ],
#         "list_meals": [
#             "list meals",
#             "show me meal options",
#             "display all available meals",
#             "اعطني قائمة الوجبات",
#             "show me foods",
#             "what meals do you have",
#             "give me the available options",
#             "list all healthy meals",
#         ],
#     }

#     for intent, templates in generic_templates.items():
#         expand_templates(templates, times=20)  # Expand each template to increase dataset size
#         for q in templates:
#             rows.append((q, intent))

#     intent_df = pd.DataFrame(rows, columns=["text", "intent"])
#     intent_df = (
#         intent_df.drop_duplicates()
#         .sample(frac=1, random_state=RANDOM_STATE)
#         .reset_index(drop=True)
#     )
#     return intent_df


# intent_df = build_intent_dataset(df)

# intent_df = balance_intents(intent_df)

# # intent_df = downsample(intent_df)



# intent_texts = intent_df["text"].tolist()
# intent_labels = intent_df["intent"].tolist()

# intent_embeddings = embedder.encode(intent_texts, convert_to_tensor=False)



# def predict_intent_semantic(user_input, threshold=0.55):
#     emb = embedder.encode([user_input])
#     sims = cosine_similarity(emb, intent_embeddings)[0]

#     idx = np.argmax(sims)
#     score = sims[idx]

#     if score < threshold:
#         return "unknown", score

#     return intent_labels[idx], score




# GOAL_KEYWORDS = {
#     "muscle gain": ["muscle gain", "gain muscle", "build muscle", "bulking", "تضخيم", "بناء عضل", "زيادة العضلات"],
#     "fat loss": ["fat loss", "weight loss", "lose fat", "cutting", "تنشيف", "رجيم", "خسارة وزن", "حرق الدهون"],
#     "maintenance": ["maintenance", "maintain weight", "stay fit", "ثبات", "المحافظة على الوزن"],
#     "endurance": ["endurance", "stamina", "energy", "تحمل", "طاقة", "قدرة تحمل"],
# }

# MEAL_TYPE_KEYWORDS = {
#     "breakfast": ["breakfast", "morning", "فطور", "إفطار"],
#     "lunch": ["lunch", "noon", "midday", "غداء"],
#     "dinner": ["dinner", "evening", "night", "عشاء"],
#     "snack": ["snack", "light meal", "between meals", "سناك", "وجبة خفيفة"],
#     "dessert": ["dessert", "sweet", "حلى", "تحلية"],
# }

# WORKOUT_KEYWORDS = {
#     "pre_workout": ["pre workout", "before gym", "before training", "قبل التمرين", "قبل الرياضة"],
#     "post_workout": ["post workout", "after gym", "after training", "بعد التمرين", "بعد الرياضة"],
# }

# def extract_goal(text):
#     text_l = normalize_text(text)
#     for goal, keys in GOAL_KEYWORDS.items():
#         if any(k in text_l for k in keys):
#             return goal
#     return None

# def extract_meal_type(text):
#     text_l = normalize_text(text)
#     for meal_type, keys in MEAL_TYPE_KEYWORDS.items():
#         if any(k in text_l for k in keys):
#             return meal_type
#     return None

# def extract_workout_intent(text):
#     text_l = normalize_text(text)
#     for wt, keys in WORKOUT_KEYWORDS.items():
#         if any(k in text_l for k in keys):
#             return wt
#     return None

# def find_meal_name_in_text(text, names=None):
#     if names is None:
#         names = df["Name"].tolist()
#     text_l = normalize_text(text)

#     # exact substring match
#     exact_matches = [name for name in names if normalize_text(name) in text_l]
#     if exact_matches:
#         return exact_matches[0]

#     # fuzzy match
#     match = difflib.get_close_matches(text, names, n=1, cutoff=0.55)
#     return match[0] if match else None


# goal_weights = {
#     "fat loss": {"calories": -0.45, "protein": 0.40, "carbs": 0.05, "fat": -0.10},
#     "muscle gain": {"calories": 0.25, "protein": 0.45, "carbs": 0.20, "fat": 0.05},
#     "endurance": {"calories": 0.20, "protein": 0.20, "carbs": 0.45, "fat": -0.05},
#     "maintenance": {"calories": 0.15, "protein": 0.25, "carbs": 0.25, "fat": 0.15},
# }

# workout_weights = {
#     "pre_workout": {"calories": 0.10, "protein": 0.15, "carbs": 0.40, "fat": -0.15},
#     "post_workout": {"calories": 0.15, "protein": 0.45, "carbs": 0.25, "fat": -0.10},
# }

# def nutrition_score(frame, goal=None, workout=None):
#     cal = frame["Calories_norm"]
#     pro = frame["ProteinContent_norm"]
#     carb = frame["CarbohydrateContent_norm"]
#     fat = frame["FatContent_norm"]

#     if workout in workout_weights:
#         w = workout_weights[workout]
#     elif goal in goal_weights:
#         w = goal_weights[goal]
#     else:
#         w = goal_weights["maintenance"]

#     score = (
#         w["calories"] * cal +
#         w["protein"] * pro +
#         w["carbs"] * carb +
#         w["fat"] * fat
#     )
#     return score

# profile_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
# profile_matrix = profile_vectorizer.fit_transform(df["meal_profile"])

# def recommend_meals(query, top_n=5, goal=None, meal_type=None, exclude_names=None):
#     if exclude_names is None:
#         exclude_names = set()
#     else:
#         exclude_names = set(exclude_names)

#     q_vec = profile_vectorizer.transform([query])
#     sim = cosine_similarity(q_vec, profile_matrix).ravel()

#     frame = df.copy()
#     frame["similarity"] = sim

#     extracted_goal = goal or extract_goal(query)
#     extracted_meal_type = meal_type or extract_meal_type(query)
#     extracted_workout = extract_workout_intent(query)

#     frame["nutrition_score"] = nutrition_score(frame, extracted_goal, extracted_workout)

#     # Bonuses
#     frame["bonus"] = 0.0
#     if extracted_goal:
#         frame.loc[frame["goal_l"] == extracted_goal, "bonus"] += 0.08
#     if extracted_meal_type:
#         frame.loc[frame["meal_type_l"] == extracted_meal_type, "bonus"] += 0.12
#     if extracted_workout:
#         frame["bonus"] += 0.05

#     frame["final_score"] = 0.60 * frame["similarity"] + 0.30 * frame["nutrition_score"] + frame["bonus"]

#     if exclude_names:
#         frame = frame[~frame["Name"].isin(exclude_names)]

#     # Filter by extracted goal / meal type if available
#     if extracted_goal:
#         filtered = frame[frame["goal_l"] == extracted_goal]
#         if len(filtered) >= 3:
#             frame = filtered
#     if extracted_meal_type:
#         filtered = frame[frame["meal_type_l"] == extracted_meal_type]
#         if len(filtered) >= 3:
#             frame = filtered

#     result = frame.sort_values("final_score", ascending=False).head(top_n).copy()
#     cols = ["Name", "Goal", "Meal Type", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent", "final_score"]
#     return result[cols].reset_index(drop=True)




# def find_meal_name_in_text(text, cutoff=0.55):
#     text_n = normalize_text(text)
#     meal_names = df["Name"].dropna().tolist()


#     # 1) exact/partial substring match
#     exact_matches = []
#     for name in meal_names:
#         name_n = normalize_text(name)
#         if name_n in text_n:
#             exact_matches.append(name)

#     if len(exact_matches) == 1:
#         return exact_matches[0]
#     if len(exact_matches) > 1:
#         # If multiple exact matches, pick the longest one (most specific)
#         return exact_matches
#     # 2) fuzzy match on full names
#     fuzzy_matches = difflib.get_close_matches(text_n, meal_names, n=3, cutoff=cutoff)
#     if fuzzy_matches:
#         return fuzzy_matches[0] if len(fuzzy_matches) == 1 else fuzzy_matches
    

#     # 3) token overlap fallback
#     text_tokens = set(text_n.split())
#     scored = []
#     for name in meal_names:
#         name_tokens = set(normalize_text(name).split())
#         overlap = len(text_tokens & name_tokens)
#         if overlap > 0:
#             score = overlap / len(name_tokens)
#             scored.append((score, name))

#     if scored:
#         scored.sort(reverse=True)
#         best_score = scored[0][0]
#         best = [name for score, name in scored if score == best_score]
#         return best[0] if len(best) == 1 else best

#     return None 

# def get_meal_row(meal_name):
#     match = df[df["Name"].str.lower() == normalize_text(meal_name)]
#     if len(match) > 0:
#         return match.iloc[0]
#     # fallback fuzzy
#     candidates = difflib.get_close_matches(meal_name, df["Name"].tolist(), n=1, cutoff=0.55)
#     if candidates:
#         return df[df["Name"] == candidates[0]].iloc[0]
#     return None

# def analyze_meal(meal_name, style="structured"):
#     row = get_meal_row(meal_name)
#     if row is None:
#         return {"error": f"Meal not found: {meal_name}"}
#     result = {
#         "Name": row["Name"],
#         "Goal": row["Goal"],
#         "Meal Type": row["Meal Type"],
#         # "Cuisine": row["Cuisine"],
#         "Calories": int(row["Calories"]),
#         "ProteinContent": int(row["ProteinContent"]),
#         "CarbohydrateContent": int(row["CarbohydrateContent"]),
#         "FatContent": int(row["FatContent"]),
#         "Ingredients": row["Ingredients"],
#     }

#     if style == "colloquial":
#         return format_analysis_colloquial(row)
    
#     return result

# def substitute_meal(meal_name, top_n=5, style="structured"):
#     row = get_meal_row(meal_name)
#     if row is None:
#         error_df = pd.DataFrame([{"error": f"Meal not found: {meal_name}"}])
#         return f"I couldn't find the meal '{meal_name}'. Please check the name and try again." if style == "colloquial" else error_df

#     query = row["meal_profile"]
#     q_vec = profile_vectorizer.transform([query])
#     sim = cosine_similarity(q_vec, profile_matrix).ravel()

#     frame = df.copy()
#     frame["similarity"] = sim
#     frame = frame[frame["Name"] != row["Name"]]
#     frame["nutrition_score"] = nutrition_score(frame, goal=row["goal_l"])
#     frame["final_score"] = 0.75 * frame["similarity"] + 0.25 * frame["nutrition_score"]

#     # Prefer same goal and meal type if possible
#     same_goal = frame[frame["goal_l"] == row["goal_l"]]
#     if len(same_goal) >= 3:
#         frame = same_goal

#     same_type = frame[frame["meal_type_l"] == row["meal_type_l"]]
#     if len(same_type) >= 3:
#         frame = same_type

#     cols = ["Name", "Goal", "Meal Type", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent", "final_score"]
#     result = frame.sort_values("final_score", ascending=False).head(top_n)[cols].reset_index(drop=True)

#     if style == "colloquial":
#         return format_substitutes_colloquial(meal_name, result)
#     return result

    

# def format_analysis_colloquial(row):
#     return (
#         f"Sure, the meal '{row['Name']}' is a {row['Meal Type']} suitable for {row['Goal']}.\n"
#         f"It is suitable for the goal of {row['Goal']} because it has {row['Calories']} calories, "
#         f"{row['ProteinContent']}g of protein, {row['CarbohydrateContent']}g of carbohydrates, and "
#         f"{row['FatContent']}g of fat."
#         f"In short: It is a good meal if you want a clear, balanced choice with basic ingredients: {row['Ingredients']}."
#     )


# def format_substitutes_colloquial(meal_name, substitutes_df):
#     if substitutes_df is None or len(substitutes_df) == 0:
#         return f"Sorry, I couldn't find any substitutes for '{meal_name}'."

#     lines = [f"Here are some healthier substitutes for '{meal_name}':"]
#     for i, (_, row) in enumerate(substitutes_df.iterrows(), start=1):
#         lines.append(
#             f"{i}) **{row['Name']}** — {row['Calories']} calories "
#             f"protein: {row['ProteinContent']}g, carbohydrates: {row['CarbohydrateContent']}g, fat: {row['FatContent']}g. "
#             f"[type: {row['Meal Type']}]"
#         )

#     lines.append("In short:These are good options if you want something similar but in a different order or a more relaxed approach.")


#     return "\n".join(lines)


# def compare_meals(meal_a, meal_b, goal=None):
#     # print(f"Comparing '{meal_a}' vs '{meal_b}' for goal: {goal}")
#     row_a = get_meal_row(meal_a)
#     row_b = get_meal_row(meal_b)
#     # print("Row A:", row_a["Name"] if row_a is not None else "Not found")
#     # print("Row B:", row_b["Name"] if row_b is not None else "Not found")


#     if row_a is None or row_b is None:
#         return {"error": "One or both meals were not found."}

#     goal = goal or "maintenance"

#     df_pair = pd.DataFrame([row_a, row_b]).copy()
#     df_pair["nutrition_score"] = nutrition_score(df_pair, goal=goal)

#     winner_idx = df_pair["nutrition_score"].idxmax()
#     winner = df_pair.loc[winner_idx, "Name"]

#     return {
#         "meal_a": row_a[["Name", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]].to_dict(),
#         "meal_b": row_b[["Name", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]].to_dict(),
#         "goal": goal,
#         "winner_for_goal": winner
#     }

# def daily_plan(goal="maintenance"):
#     meals = {}
#     meals["breakfast"] = recommend_meals(f"breakfast for {goal}", top_n=1, goal=goal, meal_type="breakfast")
#     meals["lunch"] = recommend_meals(f"lunch for {goal}", top_n=1, goal=goal, meal_type="lunch")
#     meals["dinner"] = recommend_meals(f"dinner for {goal}", top_n=1, goal=goal, meal_type="dinner")
#     meals["snack"] = recommend_meals(f"snack for {goal}", top_n=1, goal=goal, meal_type="snack")
#     return meals

# def weekly_plan(goal="maintenance"):
#     days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#     plan = []
#     used = set()
#     for day in days:
#         b = recommend_meals(f"breakfast for {goal}", top_n=1, goal=goal, meal_type="breakfast", exclude_names=used)
#         l = recommend_meals(f"lunch for {goal}", top_n=1, goal=goal, meal_type="lunch", exclude_names=used)
#         d = recommend_meals(f"dinner for {goal}", top_n=1, goal=goal, meal_type="dinner", exclude_names=used)
#         if len(b): used.add(b.iloc[0]["Name"])
#         if len(l): used.add(l.iloc[0]["Name"])
#         if len(d): used.add(d.iloc[0]["Name"])
#         plan.append({
#             "Day": day,
#             "Breakfast": b.iloc[0]["Name"] if len(b) else None,
#             "Lunch": l.iloc[0]["Name"] if len(l) else None,
#             "Dinner": d.iloc[0]["Name"] if len(d) else None,
#         })
#     return pd.DataFrame(plan)



# class FeedbackBandit:
#     def __init__(self, epsilon=0.15):
#         self.epsilon = epsilon
#         self.counts = defaultdict(int)
#         self.values = defaultdict(float)

#     def choose(self, candidates):
#         if not candidates:
#             return None
#         if random.random() < self.epsilon:
#             return random.choice(candidates)
#         return max(candidates, key=lambda x: self.values[x])

#     def update(self, meal_name, reward):
#         self.counts[meal_name] += 1
#         n = self.counts[meal_name]
#         old = self.values[meal_name]
#         self.values[meal_name] = old + (reward - old) / n

# bandit = FeedbackBandit(epsilon=0.15)

# def recommend_with_feedback(query, top_n=5):
#     recs = recommend_meals(query, top_n=top_n)
#     if recs.empty:
#         return recs
#     chosen = bandit.choose(recs["Name"].tolist())
#     return recs, chosen




# class ChatMemory:
#     def __init__(self):
#         self.last_goal = None
#         self.last_meal_type = None
#         self.last_meal_name = None
#         self.last_compare_meals = None
#         self.pending_intent = None
#         self.pending_options = None

#     def update(self, intent=None, goal=None, meal_type=None, meal_name=None,
#                compare_meals=None, pending_intent=None, pending_options=None):
#         if goal is not None:
#             self.last_goal = goal
#         if meal_type is not None:
#             self.last_meal_type = meal_type
#         if meal_name is not None:
#             self.last_meal_name = meal_name
#         if compare_meals is not None:
#             self.last_compare_meals = compare_meals
#         if pending_intent is not None:
#             self.pending_intent = pending_intent
#         if pending_options is not None:
#             self.pending_options = pending_options


# memory = ChatMemory()




# def ask_clarification(goal=None, meal_type=None):
#     return {
#         "intent": "clarification",
#         "question": "Do you want breakfast, lunch, or dinner?",
#         "goal": goal,
#         "meal_type": meal_type
#     }



# def chatbot(user_text):

#     # 1. Intent
#     intent, score = predict_intent_semantic(user_text)
#     # print(f"Predicted intent: {intent} (score: {score:.2f})")
#     # intent = intent_model.predict([user_text])[0]

#     # 2. 🤝 Greeting
#     greeting_msg = None
#     if is_greeting(text=user_text):
#         if intent != "unknown":
#             greeting_msg = handle_greeting(intent)["greeting"]
#         else:
#             return handle_greeting()

#     # 3. Extract slots
#     goal = extract_goal(user_text) or memory.last_goal
#     meal_type = extract_meal_type(user_text) or memory.last_meal_type
#     workout = extract_workout_intent(user_text)
#     meal_name = find_meal_name_in_text(user_text) or memory.last_meal_name
   
   
#     pending_intent = memory.pending_intent
#     pending_options = memory.pending_options
#     last_compare = memory.last_compare_meals

#     # 🔥 HANDLE PENDING CLARIFICATION
#     if pending_intent == "compare_meals" and pending_options:
#         selected = find_meal_name_in_text(user_text)

#         # إذا المستخدم اختار وجبة من الخيارات
#         if isinstance(selected, str) and selected in pending_options:
#             # إذا كان عندنا وجبة واحدة محفوظة سابقاً
#             if last_compare and len(last_compare) == 1:
#                 meal_a = last_compare[0]
#                 meal_b = selected
#             else:
#                 meal_a = pending_options[0]
#                 meal_b = selected

#             # تنظيف الذاكرة
#             memory.update(
#                 pending_intent=None,
#                 pending_options=None,
#                 compare_meals=[meal_a, meal_b]
#             )

#             result = compare_meals(meal_a, meal_b, goal=goal)

#             return {
#                 "intent": "compare_meals",
#                 "comparison": result
#             }
        

    
#     # 4. Routing
#     if intent == "recommend_meal":
#         if not goal or not meal_type:
#             return ask_clarification(goal=goal, meal_type=meal_type)

#         top = recommend_meals(user_text, top_n=1, goal=goal, meal_type=meal_type)
#         memory.update(intent, goal, meal_type)
#         response = {
#             "intent": intent,
#             "goal": goal,
#             "meal_type": meal_type,
#             "results": top,
#         }
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "list_meals":
#         # list meals by inferred meal type or goal
#         query = user_text
#         top = recommend_meals(query, top_n=10, goal=goal, meal_type=meal_type)
#         response = {
#             "intent": intent,
#             "goal": goal,
#             "meal_type": meal_type,
#             "results": top,
#         }
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "pre_workout":
#         top = recommend_meals(
#             user_text,
#             top_n=5,
#             workout="pre_workout" if False else None,
#             meal_type=meal_type,
#         )
#         # use nutrition scorer through query text
#         response = {"intent": intent, "results": top}
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "post_workout":
#         top = recommend_meals(user_text, top_n=5, meal_type=meal_type)
#         response = {"intent": intent, "results": top}
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "daily_plan":
#         g = goal or "maintenance"
#         response = {"intent": intent, "goal": g, "plan": daily_plan(g)}
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "weekly_plan":
#         g = goal or "maintenance"
#         response = {"intent": intent, "goal": g, "plan": weekly_plan(g)}
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "analyze_meal":
#         if meal_name is None:
#             response = {"intent": intent, "error": "Please mention a meal name."}
#             if greeting_msg:
#                 response["greeting"] = greeting_msg
#             return response

#         # row = df[df["Name"].str.lower() == meal_name].iloc[0]
#         memory.update(intent, meal_name=meal_name)

#         response = {
#             "intent": intent,
#             "analysis": analyze_meal(meal_name, style="colloquial"),
#         }
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "substitute_meal":
#         if meal_name is None:
#             response = {"intent": intent, "error": "Please mention a meal name."}
#             if greeting_msg:
#                 response["greeting"] = greeting_msg
#             return response

#         if isinstance(meal_name, list):
#             response = {
#                 "intent": intent,
#                 "error": "I found multiple matching meals. Please specify one.",
#                 "candidates": meal_name[:5],
#             }
#             if greeting_msg:
#                 response["greeting"] = greeting_msg
#             return response

#         response = {
#             "intent": intent,
#             "substitutes": substitute_meal(meal_name, top_n=5, style="colloquial"),
#         }
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     if intent == "compare_meals":
#         names = []

#         for m in df["Name"].tolist():
#             if normalize_text(m) in normalize_text(user_text):
#                 if m not in names:
#                     names.append(m)
#         # fallback split
#         parts = re.split(r"\band\b", user_text, flags=re.IGNORECASE)
#         for p in parts:
#             m = find_meal_name_in_text(p)
#             if m and m not in names:
#                 names.append(m)

#         if len(names) < 2:
#             response = {
#                 "intent": intent,
#                 "error": "Please mention 2 meal names clearly.",
#             }
#             if greeting_msg:
#                 response["greeting"] = greeting_msg
#             return response
        

        
#         # 🔴 clarification step
#         if isinstance(names[0], list) or isinstance(names[1], list):
            
#             options = []
#             matched_compare_meals = [] = []

#             if isinstance(names[0], list):
#                 options.extend(names[0][:5])
#             if isinstance(names[1], list):
#                 options.extend(names[1][:5])

#             if isinstance(names[0], str):
#                 matched_compare_meals.append(names[0])
#             if isinstance(names[1], str):
#                 matched_compare_meals.append(names[1])

#             response = {
#                 "intent": "clarification",
#                 "message": "I couldn't clearly identify two meals.",
#                 "question": "Which meals do you want to compare?",
#                 "options": options,
#             }

#             memory.update(
#                 intent="compare_meals",
#                 compare_meals=matched_compare_meals,
#                 pending_intent="compare_meals",
#                 pending_options=options
#             )

#             if greeting_msg:
#                 response["greeting"] = greeting_msg

#             return response

#         result = compare_meals(names[0], names[1], goal=goal)
#         memory.update(
#             intent="compare_meals",
#             compare_meals=[names[0], names[1]],
#             meal_name=names[1]
#         )

#         if "error" in result:
#             return {"intent": intent, "message": result["error"]}

#         return {"intent": intent, "comparison": result}

#     if intent == "unknown":
#         response = {
#             "intent": "unknown",
#             "message": "I didn't quite understand you 😅 Could you rephrase it?",
#         }
#         if greeting_msg:
#             response["greeting"] = greeting_msg
#         return response

#     # return "OK 👍"
