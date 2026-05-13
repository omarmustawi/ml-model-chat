import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from core.config import DATA_PATH

from utils.text_utils import normalize_text

from constants.keywords import GOAL_SYNONYMS, MEAL_SYNONYMS

from sentence_transformers import SentenceTransformer



RANDOM_STATE = 42 


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


# preprocessing هنا

# تنظيف البيانات
for col in ["Goal", "Meal Type", "Name"]:
    df[col] = df[col].astype(str)

df["goal_l"] = df["Goal"].apply(normalize_text)
df["meal_type_l"] = df["Meal Type"].apply(normalize_text)
df["name_l"] = df["Name"].apply(normalize_text)
df["ingredients_l"] = df["Ingredients"].apply(normalize_text)

# إنشاء meal_profile
df["meal_profile"] = (
    "name: "
    + df["Name"].astype(str)
    + " | goal: "
    + df["Goal"].astype(str)
    + " | meal type: "
    + df["Meal Type"].astype(str)
    + " | ingredients: "
    + df["Ingredients"].astype(str)
)


numeric_cols = [
    "Calories",
    "ProteinContent",
    "CarbohydrateContent",
    "FatContent",
]

for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()

    df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)



id="fix1"
from sklearn.utils import resample

def balance_intents(df):
    max_count = df['intent'].value_counts().max()
    balanced = []

    for intent in df['intent'].unique():
        subset = df[df['intent'] == intent]

        if len(subset) < max_count:
            subset = resample(subset,
                             replace=True,
                             n_samples=max_count,
                             random_state=42)

        balanced.append(subset)

    return pd.concat(balanced)



id="fix2"
def downsample(df, max_per_class=200):
    parts = []
    for intent in df['intent'].unique():
        subset = df[df['intent'] == intent]
        subset = subset.sample(min(len(subset), max_per_class), random_state=42)
        parts.append(subset)
    return pd.concat(parts)


id="fix3"
def expand_templates(templates, times=20):
    new = []
    for _ in range(times):
        new.extend(templates)
    return new




def generate_goal_queries(goal):
    synonyms = GOAL_SYNONYMS.get(goal.lower(), [goal])
    meal_types = ["breakfast", "lunch", "dinner", "snack", "dessert"]
    queries = []
    for syn in synonyms:
        queries.extend(
            [
                f"recommend a meal for {syn}",
                f"give me a healthy meal for {syn}",
                f"I need food for {syn}",
                f"اقتراح وجبة لـ {syn}",
                f"اعطني وجبة مناسبة لـ {syn}",
            ]
        )
        for mt in meal_types:
            queries.extend(
                [
                    f"recommend a {mt} for {syn}",
                    f"give me a {mt} meal for {syn}",
                    f"show me {mt} options for {syn}",
                ]
            )
    return queries


def generate_mealtype_queries(meal_type):
    synonyms = MEAL_SYNONYMS.get(meal_type.lower(), [meal_type])
    queries = []
    for syn in synonyms:
        queries.extend(
            [
                f"list {syn} meals",
                f"show me {syn} options",
                f"I want a {syn}",
                f"اعطني وجبة {syn}",
                f"أريد {syn} صحي",
            ]
        )
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
        rows.extend(
            [
                (f"analyze {meal_name}", "analyze_meal"),
                (f"what are the calories and macros of {meal_name}", "analyze_meal"),
                (f"compare {meal_name} with another meal", "compare_meals"),
                (f"suggest a substitute for {meal_name}", "substitute_meal"),
            ]
        )

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
        expand_templates(templates, times=20)  # Expand each template to increase dataset size
        for q in templates:
            rows.append((q, intent))

    intent_df = pd.DataFrame(rows, columns=["text", "intent"])
    intent_df = (
        intent_df.drop_duplicates()
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return intent_df


intent_df = build_intent_dataset(df)

intent_df = balance_intents(intent_df)



intent_texts = intent_df["text"].tolist()
intent_labels = intent_df["intent"].tolist()

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


intent_embeddings = embedder.encode(intent_texts, convert_to_tensor=False)


profile_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
profile_matrix = profile_vectorizer.fit_transform(df["meal_profile"])



print("end of preprocessing ===================================")