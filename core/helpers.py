import re
import difflib

from core.preprocessing import df


from constants.keywords import GOAL_KEYWORDS, MEAL_TYPE_KEYWORDS, GOAL_KEYWORDS, WORKOUT_KEYWORDS, GREETINGS
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
        "greeting": "👋 How can I help you? You can request a meal, meal analysis, or a diet plan."
    }

