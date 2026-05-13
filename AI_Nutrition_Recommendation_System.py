# %% [markdown]
"""
# AI Nutrition Recommendation System

هذا الـ notebook يبني مشروع تخرج قوي ومناسب لعرضه كمشروع AI عملي:

- فهم نية المستخدم (Intent Classification)
- اقتراح وجبات ذكي حسب الهدف ونوع الوجبة
- تحليل الوجبة وتقدير السعرات والماكروز
- خطة يومية وأسبوعية
- جزء تعلّم من التغذية الراجعة بشكل بسيط يشبه Reinforcement Learning
"""

# %%

# Cell 1 — Imports and configuration
import os
import re
import json
import math
import random
import joblib
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 50)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_PATH = "arab_meals_120.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# %%

# Cell 2 — Load data
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
display(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

# %%

# Cell 3 — Clean and standardize columns
df = df.copy()

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

print("After cleaning:", df.shape)
df.head()

# %%

# Cell 4 — Quick EDA
print("Unique goals:", df["Goal"].unique())
print("Unique meal types:", df["Meal Type"].unique())
print("\nGoal counts:\n", df["Goal"].value_counts())
print("\nMeal type counts:\n", df["Meal Type"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df["Goal"].value_counts().sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Goal Distribution")
axes[0].set_xlabel("Count")

df["Meal Type"].value_counts().sort_values().plot(kind="barh", ax=axes[1])
axes[1].set_title("Meal Type Distribution")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.show()

# %%

# Cell 5 — Build text fields for modeling
def normalize_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())

df["goal_l"] = df["Goal"].apply(normalize_text)
df["meal_type_l"] = df["Meal Type"].apply(normalize_text)
df["name_l"] = df["Name"].apply(normalize_text)
df["ingredients_l"] = df["Ingredients"].apply(normalize_text)
df["cuisine_l"] = df["Cuisine"].apply(normalize_text)

df["meal_profile"] = (
    "name: " + df["Name"].astype(str) +
    " | goal: " + df["Goal"].astype(str) +
    " | meal type: " + df["Meal Type"].astype(str) +
    " | cuisine: " + df["Cuisine"].astype(str) +
    " | ingredients: " + df["Ingredients"].astype(str)
)

# Normalized nutrition columns for ranking
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)

df[["meal_profile"] + numeric_cols].head()

# %% [markdown]
"""
## 1) Intent Classifier

هنا بنولّد بيانات أسئلة صناعية من الداتا نفسها، ثم ندرب موديل يفهم نوع الطلب:
- اقتراح وجبة
- قائمة وجبات
- خطة يومية
- خطة أسبوعية
- قبل التمرين
- بعد التمرين
- تحليل وجبة
- مقارنة وجبتين
- بدائل وجبة
"""

# %%

# Cell 6 — Synthetic intent dataset
GOAL_SYNONYMS = {
    "muscle gain": ["muscle gain", "gain muscle", "build muscle", "bulking", "increase muscle", "تضخيم", "بناء عضل", "زيادة العضلات"],
    "fat loss": ["fat loss", "lose fat", "weight loss", "cutting", "lose weight", "تنشيف", "خسارة وزن", "رجيم", "حرق الدهون"],
    "maintenance": ["maintenance", "maintain weight", "stay fit", "ثبات", "المحافظة على الوزن"],
    "endurance": ["endurance", "stamina", "energy", "athletic endurance", "تحمل", "طاقة", "قدرة تحمل"],
}

MEAL_SYNONYMS = {
    "breakfast": ["breakfast", "morning meal", "first meal", "فطور", "إفطار"],
    "lunch": ["lunch", "midday meal", "noon meal", "غداء"],
    "dinner": ["dinner", "evening meal", "night meal", "عشاء"],
    "snack": ["snack", "light meal", "between meals", "سناك", "وجبة خفيفة"],
    "dessert": ["dessert", "sweet", "after meal", "حلى", "تحلية"],
}

INTENT_LABELS = [
    "recommend_meal",
    "list_meals",
    "daily_plan",
    "weekly_plan",
    "pre_workout",
    "post_workout",
    "analyze_meal",
    "compare_meals",
    "substitute_meal",
]

def generate_goal_queries(goal):
    synonyms = GOAL_SYNONYMS.get(goal.lower(), [goal])
    meal_types = ["breakfast", "lunch", "dinner", "snack", "dessert"]
    queries = []
    for syn in synonyms:
        queries.extend([
            f"recommend a meal for {syn}",
            f"give me a healthy meal for {syn}",
            f"I need food for {syn}",
            f"اقتراح وجبة لـ {syn}",
            f"اعطني وجبة مناسبة لـ {syn}",
        ])
        for mt in meal_types:
            queries.extend([
                f"recommend a {mt} for {syn}",
                f"give me a {mt} meal for {syn}",
                f"show me {mt} options for {syn}",
            ])
    return queries

def generate_mealtype_queries(meal_type):
    synonyms = MEAL_SYNONYMS.get(meal_type.lower(), [meal_type])
    queries = []
    for syn in synonyms:
        queries.extend([
            f"list {syn} meals",
            f"show me {syn} options",
            f"I want a {syn}",
            f"اعطني وجبة {syn}",
            f"أريد {syn} صحي",
        ])
    return queries

def build_intent_dataset(df):
    rows = []

    # Goal-based and meal-type-based recommendation/listing examples
    for _, row in df.iterrows():
        goal = row["Goal"].lower()
        meal_type = row["Meal Type"].lower()
        meal_name = row["Name"]

        # recommend_meal
        for q in generate_goal_queries(goal)[:8]:
            rows.append((q, "recommend_meal"))
        for q in generate_mealtype_queries(meal_type)[:6]:
            rows.append((q, "list_meals"))

        # meal-specific intents
        rows.extend([
            (f"analyze {meal_name}", "analyze_meal"),
            (f"what are the calories and macros of {meal_name}", "analyze_meal"),
            (f"compare {meal_name} with another meal", "compare_meals"),
            (f"suggest a substitute for {meal_name}", "substitute_meal"),
        ])

    # Generic plan / workout intents — multiple templates for each class
    generic_templates = {
        "daily_plan": [
            "create a daily meal plan for fat loss",
            "make me a one day meal plan for muscle gain",
            "daily nutrition plan for maintenance",
            "اعمل لي خطة يومية للتنشيف",
            "plan my meals for today for endurance",
            "give me a full day meal plan",
            "build a daily nutrition plan for me",
            "one day food plan for weight loss",
        ],
        "weekly_plan": [
            "create a weekly meal plan for fat loss",
            "make me a 7 day meal plan for muscle gain",
            "weekly nutrition plan for maintenance",
            "اعمل لي خطة أسبوعية للتضخيم",
            "plan my meals for the week for endurance",
            "give me a 7 day nutrition plan",
            "build a weekly food plan",
            "one week diet plan for fitness",
        ],
        "pre_workout": [
            "recommend a pre workout meal",
            "what should I eat before training",
            "وجبة قبل التمرين",
            "اقتراح سناك قبل التمرين",
            "best meal before gym",
            "what to eat before workout",
            "need energy before training",
            "suggest a light pre training meal",
        ],
        "post_workout": [
            "recommend a post workout meal",
            "what should I eat after training",
            "وجبة بعد التمرين",
            "اقتراح وجبة بعد التمرين",
            "best meal after gym",
            "what to eat after workout",
            "need recovery food after training",
            "suggest a high protein post training meal",
        ],
        "compare_meals": [
            "compare two meals for me",
            "which meal is better for fat loss",
            "قارن بين وجبتين",
            "أي وجبة أفضل للتضخيم",
            "compare meal A and meal B",
            "which option has better macros",
            "help me compare foods",
            "which is healthier between these meals",
        ],
        "analyze_meal": [
            "analyze this meal",
            "show me nutrition facts",
            "ما هي القيم الغذائية لهذه الوجبة",
            "احسب السعرات والماكروز",
            "what are the calories in this meal",
            "show protein carbs and fat",
            "nutrition breakdown for this food",
            "meal calorie analysis",
        ],
        "substitute_meal": [
            "suggest a healthy substitution",
            "replace this meal with a better option",
            "اعطني بديل صحي",
            "اقترح بديل لوجبة",
            "find a similar healthier meal",
            "recommend a substitute with similar taste",
            "give me an alternative food",
            "swap this meal with a lighter option",
        ],
        "recommend_meal": [
            "recommend a meal",
            "suggest me food",
            "اعطني وجبة مناسبة",
            "اقتراح وجبة صحية",
            "give me something to eat",
            "pick a meal for me",
            "show me a good meal option",
            "I want a meal suggestion",
        ],
        "list_meals": [
            "list meals",
            "show me meal options",
            "display all available meals",
            "اعطني قائمة الوجبات",
            "show me foods",
            "what meals do you have",
            "give me the available options",
            "list all healthy meals",
        ],
    }

    for intent, templates in generic_templates.items():
        for q in templates:
            rows.append((q, intent))

    intent_df = pd.DataFrame(rows, columns=["text", "intent"])
    intent_df = intent_df.drop_duplicates().sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return intent_df

intent_df = build_intent_dataset(df)
print("Intent dataset shape:", intent_df.shape)
display(intent_df.head(10))
print(intent_df["intent"].value_counts())

# %%

# Cell 7 — Train intent classifier
X_train, X_test, y_train, y_test = train_test_split(
    intent_df["text"],
    intent_df["intent"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=intent_df["intent"]
)

intent_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])

intent_model.fit(X_train, y_train)
pred = intent_model.predict(X_test)

print("Intent Accuracy:", accuracy_score(y_test, pred))
print("\nClassification report:\n")
print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred, labels=INTENT_LABELS)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(INTENT_LABELS)))
ax.set_yticks(range(len(INTENT_LABELS)))
ax.set_xticklabels(INTENT_LABELS, rotation=45, ha="right")
ax.set_yticklabels(INTENT_LABELS)
ax.set_title("Intent Confusion Matrix")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 2) Nutrition Prediction Model

هذا الموديل يتعلم العلاقة بين وصف الوجبة ومحتواها الغذائي:
- Calories
- Protein
- Carbs
- Fat

هذا الجزء مفيد جداً في العرض لأنه يثبت أن عندك موديل تنبؤ حقيقي وليس فقط قواعد.
"""

# %%

# Cell 8 — Train nutrition regressor
nutrition_features = ["meal_profile"]
target_cols = ["Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]

X = df["meal_profile"]
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

nutrition_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
nutrition_svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)

X_train_vec = nutrition_vectorizer.fit_transform(X_train)
X_test_vec = nutrition_vectorizer.transform(X_test)

X_train_red = nutrition_svd.fit_transform(X_train_vec)
X_test_red = nutrition_svd.transform(X_test_vec)

nutrition_model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=RANDOM_STATE), n_jobs=1)
nutrition_model.fit(X_train_red, y_train)

y_pred = nutrition_model.predict(X_test_red)

for i, col in enumerate(target_cols):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col}: MAE={mae:.2f} | R2={r2:.3f}")

# Show a few predictions
pred_df = y_test.copy().reset_index(drop=True)
for i, col in enumerate(target_cols):
    pred_df[f"{col}_pred"] = np.round(y_pred[:, i], 1)

display(pred_df.head(10))

# %% [markdown]
"""
## 3) Recommendation Engine

الترشيح هنا هجين:
- تشابه نصي مع طلب المستخدم
- مطابقة الهدف ونوع الوجبة
- درجة غذائية حسب الهدف
- دعم قبل/بعد التمرين
"""

# %%

# Cell 9 — Helpers for parsing user requests
GOAL_KEYWORDS = {
    "muscle gain": ["muscle gain", "gain muscle", "build muscle", "bulking", "تضخيم", "بناء عضل", "زيادة العضلات"],
    "fat loss": ["fat loss", "weight loss", "lose fat", "cutting", "تنشيف", "رجيم", "خسارة وزن", "حرق الدهون"],
    "maintenance": ["maintenance", "maintain weight", "stay fit", "ثبات", "المحافظة على الوزن"],
    "endurance": ["endurance", "stamina", "energy", "تحمل", "طاقة", "قدرة تحمل"],
}

MEAL_TYPE_KEYWORDS = {
    "breakfast": ["breakfast", "morning", "فطور", "إفطار"],
    "lunch": ["lunch", "noon", "midday", "غداء"],
    "dinner": ["dinner", "evening", "night", "عشاء"],
    "snack": ["snack", "light meal", "between meals", "سناك", "وجبة خفيفة"],
    "dessert": ["dessert", "sweet", "حلى", "تحلية"],
}

WORKOUT_KEYWORDS = {
    "pre_workout": ["pre workout", "before gym", "before training", "قبل التمرين", "قبل الرياضة"],
    "post_workout": ["post workout", "after gym", "after training", "بعد التمرين", "بعد الرياضة"],
}

def extract_goal(text):
    text_l = normalize_text(text)
    for goal, keys in GOAL_KEYWORDS.items():
        if any(k in text_l for k in keys):
            return goal
    return None

def extract_meal_type(text):
    text_l = normalize_text(text)
    for meal_type, keys in MEAL_TYPE_KEYWORDS.items():
        if any(k in text_l for k in keys):
            return meal_type
    return None

def extract_workout_intent(text):
    text_l = normalize_text(text)
    for wt, keys in WORKOUT_KEYWORDS.items():
        if any(k in text_l for k in keys):
            return wt
    return None

def find_meal_name_in_text(text, names=None):
    if names is None:
        names = df["Name"].tolist()
    text_l = normalize_text(text)

    # exact substring match
    exact_matches = [name for name in names if normalize_text(name) in text_l]
    if exact_matches:
        return exact_matches[0]

    # fuzzy match
    match = difflib.get_close_matches(text, names, n=1, cutoff=0.55)
    return match[0] if match else None

# %%

# Cell 10 — Recommendation scoring
goal_weights = {
    "fat loss": {"calories": -0.45, "protein": 0.40, "carbs": 0.05, "fat": -0.10},
    "muscle gain": {"calories": 0.25, "protein": 0.45, "carbs": 0.20, "fat": 0.05},
    "endurance": {"calories": 0.20, "protein": 0.20, "carbs": 0.45, "fat": -0.05},
    "maintenance": {"calories": 0.15, "protein": 0.25, "carbs": 0.25, "fat": 0.15},
}

workout_weights = {
    "pre_workout": {"calories": 0.10, "protein": 0.15, "carbs": 0.40, "fat": -0.15},
    "post_workout": {"calories": 0.15, "protein": 0.45, "carbs": 0.25, "fat": -0.10},
}

def nutrition_score(frame, goal=None, workout=None):
    cal = frame["Calories_norm"]
    pro = frame["ProteinContent_norm"]
    carb = frame["CarbohydrateContent_norm"]
    fat = frame["FatContent_norm"]

    if workout in workout_weights:
        w = workout_weights[workout]
    elif goal in goal_weights:
        w = goal_weights[goal]
    else:
        w = goal_weights["maintenance"]

    score = (
        w["calories"] * cal +
        w["protein"] * pro +
        w["carbs"] * carb +
        w["fat"] * fat
    )
    return score

profile_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
profile_matrix = profile_vectorizer.fit_transform(df["meal_profile"])

def recommend_meals(query, top_n=5, goal=None, meal_type=None, exclude_names=None):
    if exclude_names is None:
        exclude_names = set()
    else:
        exclude_names = set(exclude_names)

    q_vec = profile_vectorizer.transform([query])
    sim = cosine_similarity(q_vec, profile_matrix).ravel()

    frame = df.copy()
    frame["similarity"] = sim

    extracted_goal = goal or extract_goal(query)
    extracted_meal_type = meal_type or extract_meal_type(query)
    extracted_workout = extract_workout_intent(query)

    frame["nutrition_score"] = nutrition_score(frame, extracted_goal, extracted_workout)

    # Bonuses
    frame["bonus"] = 0.0
    if extracted_goal:
        frame.loc[frame["goal_l"] == extracted_goal, "bonus"] += 0.08
    if extracted_meal_type:
        frame.loc[frame["meal_type_l"] == extracted_meal_type, "bonus"] += 0.12
    if extracted_workout:
        frame["bonus"] += 0.05

    frame["final_score"] = 0.60 * frame["similarity"] + 0.30 * frame["nutrition_score"] + frame["bonus"]

    if exclude_names:
        frame = frame[~frame["Name"].isin(exclude_names)]

    # Filter by extracted goal / meal type if available
    if extracted_goal:
        filtered = frame[frame["goal_l"] == extracted_goal]
        if len(filtered) >= 3:
            frame = filtered
    if extracted_meal_type:
        filtered = frame[frame["meal_type_l"] == extracted_meal_type]
        if len(filtered) >= 3:
            frame = filtered

    result = frame.sort_values("final_score", ascending=False).head(top_n).copy()
    cols = ["Name", "Goal", "Meal Type", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent", "Cuisine", "final_score"]
    return result[cols].reset_index(drop=True)

# %%

# Cell 11 — Smart analysis, substitution, comparison
def get_meal_row(meal_name):
    match = df[df["Name"].str.lower() == normalize_text(meal_name)]
    if len(match) > 0:
        return match.iloc[0]
    # fallback fuzzy
    candidates = difflib.get_close_matches(meal_name, df["Name"].tolist(), n=1, cutoff=0.55)
    if candidates:
        return df[df["Name"] == candidates[0]].iloc[0]
    return None

def analyze_meal(meal_name):
    row = get_meal_row(meal_name)
    if row is None:
        return {"error": f"Meal not found: {meal_name}"}
    return {
        "Name": row["Name"],
        "Goal": row["Goal"],
        "Meal Type": row["Meal Type"],
        "Cuisine": row["Cuisine"],
        "Calories": int(row["Calories"]),
        "ProteinContent": int(row["ProteinContent"]),
        "CarbohydrateContent": int(row["CarbohydrateContent"]),
        "FatContent": int(row["FatContent"]),
        "Ingredients": row["Ingredients"],
    }

def substitute_meal(meal_name, top_n=5):
    row = get_meal_row(meal_name)
    if row is None:
        return pd.DataFrame([{"error": f"Meal not found: {meal_name}"}])

    query = row["meal_profile"]
    q_vec = profile_vectorizer.transform([query])
    sim = cosine_similarity(q_vec, profile_matrix).ravel()

    frame = df.copy()
    frame["similarity"] = sim
    frame = frame[frame["Name"] != row["Name"]]
    frame["nutrition_score"] = nutrition_score(frame, goal=row["goal_l"])
    frame["final_score"] = 0.75 * frame["similarity"] + 0.25 * frame["nutrition_score"]

    # Prefer same goal and meal type if possible
    same_goal = frame[frame["goal_l"] == row["goal_l"]]
    if len(same_goal) >= 3:
        frame = same_goal

    same_type = frame[frame["meal_type_l"] == row["meal_type_l"]]
    if len(same_type) >= 3:
        frame = same_type

    cols = ["Name", "Goal", "Meal Type", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent", "Cuisine", "final_score"]
    return frame.sort_values("final_score", ascending=False).head(top_n)[cols].reset_index(drop=True)

def compare_meals(meal_a, meal_b, goal=None):
    row_a = get_meal_row(meal_a)
    row_b = get_meal_row(meal_b)
    if row_a is None or row_b is None:
        return {"error": "One or both meals were not found."}

    goal = goal or "maintenance"

    df_pair = pd.DataFrame([row_a, row_b]).copy()
    df_pair["nutrition_score"] = nutrition_score(df_pair, goal=goal)

    winner_idx = df_pair["nutrition_score"].idxmax()
    winner = df_pair.loc[winner_idx, "Name"]

    return {
        "meal_a": row_a[["Name", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]].to_dict(),
        "meal_b": row_b[["Name", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]].to_dict(),
        "goal": goal,
        "winner_for_goal": winner
    }

def daily_plan(goal="maintenance"):
    meals = {}
    meals["breakfast"] = recommend_meals(f"breakfast for {goal}", top_n=1, goal=goal, meal_type="breakfast")
    meals["lunch"] = recommend_meals(f"lunch for {goal}", top_n=1, goal=goal, meal_type="lunch")
    meals["dinner"] = recommend_meals(f"dinner for {goal}", top_n=1, goal=goal, meal_type="dinner")
    meals["snack"] = recommend_meals(f"snack for {goal}", top_n=1, goal=goal, meal_type="snack")
    return meals

def weekly_plan(goal="maintenance"):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plan = []
    used = set()
    for day in days:
        b = recommend_meals(f"breakfast for {goal}", top_n=1, goal=goal, meal_type="breakfast", exclude_names=used)
        l = recommend_meals(f"lunch for {goal}", top_n=1, goal=goal, meal_type="lunch", exclude_names=used)
        d = recommend_meals(f"dinner for {goal}", top_n=1, goal=goal, meal_type="dinner", exclude_names=used)
        if len(b): used.add(b.iloc[0]["Name"])
        if len(l): used.add(l.iloc[0]["Name"])
        if len(d): used.add(d.iloc[0]["Name"])
        plan.append({
            "Day": day,
            "Breakfast": b.iloc[0]["Name"] if len(b) else None,
            "Lunch": l.iloc[0]["Name"] if len(l) else None,
            "Dinner": d.iloc[0]["Name"] if len(d) else None,
        })
    return pd.DataFrame(plan)

# %% [markdown]
"""
## 4) Reinforcement-Learning Style Feedback Layer

هذا جزء اختياري وخفيف.  
بدل ما ندرب RL حقيقي على بيانات صغيرة جدًا، نستخدم **online feedback**:
- المستخدم يعطي 👍 أو 👎
- النظام يحدث تفضيلاته تدريجيًا

هذا عملي أكثر للمشروع من RL كامل على 349 صف فقط.
"""

# %%

# Cell 12 — Simple feedback policy (bandit-style online learning)
class FeedbackBandit:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon
        self.counts = defaultdict(int)
        self.values = defaultdict(float)

    def choose(self, candidates):
        if not candidates:
            return None
        if random.random() < self.epsilon:
            return random.choice(candidates)
        return max(candidates, key=lambda x: self.values[x])

    def update(self, meal_name, reward):
        self.counts[meal_name] += 1
        n = self.counts[meal_name]
        old = self.values[meal_name]
        self.values[meal_name] = old + (reward - old) / n

bandit = FeedbackBandit(epsilon=0.15)

def recommend_with_feedback(query, top_n=5):
    recs = recommend_meals(query, top_n=top_n)
    if recs.empty:
        return recs
    chosen = bandit.choose(recs["Name"].tolist())
    return recs, chosen

# Example update:
# bandit.update("Chicken Salad", reward=1)   # user liked it
# bandit.update("Rice Pudding", reward=0)    # user disliked it

# %% [markdown]
"""
## 5) Unified Chatbot Function

هذا هو المدخل النهائي للمشروع.  
تكتب رسالة المستخدم، والنظام:
1. يحدد intent
2. يستخرج الهدف/نوع الوجبة
3. يرجع الرد المناسب
"""

# %%

# Cell 13 — Unified chatbot
def chatbot(user_text):
    intent = intent_model.predict([user_text])[0]
    goal = extract_goal(user_text)
    meal_type = extract_meal_type(user_text)
    workout = extract_workout_intent(user_text)
    meal_name = find_meal_name_in_text(user_text)

    if intent == "recommend_meal":
        top = recommend_meals(user_text, top_n=5, goal=goal, meal_type=meal_type)
        return {"intent": intent, "goal": goal, "meal_type": meal_type, "results": top}

    if intent == "list_meals":
        # list meals by inferred meal type or goal
        query = user_text
        top = recommend_meals(query, top_n=10, goal=goal, meal_type=meal_type)
        return {"intent": intent, "goal": goal, "meal_type": meal_type, "results": top}

    if intent == "pre_workout":
        top = recommend_meals(user_text, top_n=5, workout="pre_workout" if False else None, meal_type=meal_type)
        # use nutrition scorer through query text
        return {"intent": intent, "results": top}

    if intent == "post_workout":
        top = recommend_meals(user_text, top_n=5, meal_type=meal_type)
        return {"intent": intent, "results": top}

    if intent == "daily_plan":
        g = goal or "maintenance"
        return {"intent": intent, "goal": g, "plan": daily_plan(g)}

    if intent == "weekly_plan":
        g = goal or "maintenance"
        return {"intent": intent, "goal": g, "plan": weekly_plan(g)}

    if intent == "analyze_meal":
        if meal_name is None:
            return {"intent": intent, "error": "Please mention a meal name."}
        return {"intent": intent, "analysis": analyze_meal(meal_name)}

    if intent == "substitute_meal":
        if meal_name is None:
            return {"intent": intent, "error": "Please mention a meal name."}
        return {"intent": intent, "substitutes": substitute_meal(meal_name, top_n=5)}

    if intent == "compare_meals":
        names = []
        for m in df["Name"].tolist():
            if normalize_text(m) in normalize_text(user_text):
                names.append(m)
        # fallback: try fuzzy from the text, or split by "and"
        if len(names) < 2 and " and " in user_text.lower():
            parts = [p.strip() for p in re.split(r"\band\b", user_text, flags=re.IGNORECASE) if p.strip()]
            for p in parts:
                m = find_meal_name_in_text(p)
                if m and m not in names:
                    names.append(m)
        if len(names) < 2:
            return {"intent": intent, "error": "Please mention two meal names."}
        return {"intent": intent, "comparison": compare_meals(names[0], names[1], goal=goal)}

    return {"intent": intent, "message": "I could not route your request clearly."}

# Try examples
examples = [
    "Recommend a meal for muscle gain",
    "Give me a breakfast for fat loss",
    "Create a weekly meal plan for endurance",
    "Analyze Chicken Salad",
    "Suggest a substitute for Rice Pudding",
]
for ex in examples:
    out = chatbot(ex)
    print("\nUSER:", ex)
    print("INTENT:", out["intent"])
    if "results" in out:
        display(out["results"].head(3))
    elif "plan" in out:
        display(out["plan"].head())
    elif "analysis" in out:
        print(out["analysis"])
    elif "substitutes" in out:
        display(out["substitutes"].head(5))

# %% [markdown]
"""
## 6) Save Models

احفظي الموديلات حتى تستخدميها لاحقًا في API أو واجهة.
"""

# %%

# Cell 14 — Save artifacts
joblib.dump(intent_model, os.path.join(MODEL_DIR, "intent_model.joblib"))
joblib.dump(nutrition_model, os.path.join(MODEL_DIR, "nutrition_model.joblib"))
joblib.dump(nutrition_vectorizer, os.path.join(MODEL_DIR, "nutrition_vectorizer.joblib"))
joblib.dump(nutrition_svd, os.path.join(MODEL_DIR, "nutrition_svd.joblib"))
joblib.dump(profile_vectorizer, os.path.join(MODEL_DIR, "profile_vectorizer.joblib"))
joblib.dump(df, os.path.join(MODEL_DIR, "meals_dataframe.joblib"))
joblib.dump(bandit, os.path.join(MODEL_DIR, "bandit_state.joblib"))

print("Saved all artifacts in:", MODEL_DIR)

# %% [markdown]
"""
## 7) Demo

هذا الجزء تعرضيه بالدفاع:
- سؤال للموديل
- ترشيحات
- تحليل وجبة
- خطة أسبوعية
"""

# %%

# Cell 15 — Demo interactions
print(chatbot("I want a meal for fat loss and breakfast"))
print(chatbot("Create a daily meal plan for muscle gain"))
print(chatbot("Analyze the meal Chicken Salad"))

