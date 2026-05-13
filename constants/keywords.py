GREETINGS = [
    "hello",
    "hi",
    "hey",
    "مرحبا",
    "السلام",
    "السلام عليكم",
    "هلا",
    "صباح الخير",
    "مساء الخير",
]


GOAL_KEYWORDS = {
    "muscle gain": [
        "muscle gain",
        "gain muscle",
        "build muscle",
        "bulking",
        "تضخيم",
        "بناء عضل",
        "زيادة العضلات",
    ],
    "fat loss": [
        "fat loss",
        "weight loss",
        "lose fat",
        "cutting",
        "تنشيف",
        "رجيم",
        "خسارة وزن",
        "حرق الدهون",
    ],
    "maintenance": [
        "maintenance",
        "maintain weight",
        "stay fit",
        "ثبات",
        "المحافظة على الوزن",
    ],
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
    "pre_workout": [
        "pre workout",
        "before gym",
        "before training",
        "قبل التمرين",
        "قبل الرياضة",
    ],
    "post_workout": [
        "post workout",
        "after gym",
        "after training",
        "بعد التمرين",
        "بعد الرياضة",
    ],
}


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



GOAL_SYNONYMS = {
    "muscle gain": [
        "muscle gain",
        "gain muscle",
        "build muscle",
        "bulking",
        "increase muscle",
        "تضخيم",
        "بناء عضل",
        "زيادة العضلات",
    ],
    "fat loss": [
        "fat loss",
        "lose fat",
        "weight loss",
        "cutting",
        "lose weight",
        "تنشيف",
        "خسارة وزن",
        "رجيم",
        "حرق الدهون",
    ],
    "maintenance": [
        "maintenance",
        "maintain weight",
        "stay fit",
        "ثبات",
        "المحافظة على الوزن",
    ],
    "endurance": [
        "endurance",
        "stamina",
        "energy",
        "athletic endurance",
        "تحمل",
        "طاقة",
        "قدرة تحمل",
    ],
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
