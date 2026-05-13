import re
import difflib
import pandas as pd
from core.preprocessing import df

from core.config import RANDOM_STATE

from constants.keywords import (
    GOAL_KEYWORDS,
    MEAL_TYPE_KEYWORDS,
    GOAL_KEYWORDS,
    WORKOUT_KEYWORDS,
    GREETINGS,
    generic_templates,
    MEAL_SYNONYMS,
    GOAL_SYNONYMS,
)
from utils.text_utils import normalize_text


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


def is_greeting(text):
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)  # tokenize
    return any(g in words for g in GREETINGS)


def handle_greeting(intent=None):
    if intent:
        return {
            "intent": intent,
            "greeting": "Hello, ",
        }
    return {
        "intent": "greeting",
        "greeting": "👋 How can I help you? You can request a meal, meal analysis, or a diet plan.",
    }


from sklearn.utils import resample


def balance_intents(df):
    max_count = df["intent"].value_counts().max()
    balanced = []

    for intent in df["intent"].unique():
        subset = df[df["intent"] == intent]

        if len(subset) < max_count:
            subset = resample(
                subset, replace=True, n_samples=max_count, random_state=42
            )

        balanced.append(subset)

    return pd.concat(balanced)



def downsample(df, max_per_class=200):
    parts = []
    for intent in df["intent"].unique():
        subset = df[df["intent"] == intent]
        subset = subset.sample(min(len(subset), max_per_class), random_state=42)
        parts.append(subset)
    return pd.concat(parts)


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

    for intent, templates in generic_templates.items():
        expand_templates(
            templates, times=20
        )  # Expand each template to increase dataset size
        for q in templates:
            rows.append((q, intent))

    intent_df = pd.DataFrame(rows, columns=["text", "intent"])
    intent_df = (
        intent_df.drop_duplicates()
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return intent_df
