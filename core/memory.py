class ChatMemory:
    def __init__(self):
        self.last_goal = None
        self.last_meal_type = None
        self.last_meal_name = None
        self.last_compare_meals = None
        self.pending_intent = None
        self.pending_options = None

    def update(self, intent=None, goal=None, meal_type=None, meal_name=None,
               compare_meals=None, pending_intent=None, pending_options=None):
        if goal is not None:
            self.last_goal = goal
        if meal_type is not None:
            self.last_meal_type = meal_type
        if meal_name is not None:
            self.last_meal_name = meal_name
        if compare_meals is not None:
            self.last_compare_meals = compare_meals
        if pending_intent is not None:
            self.pending_intent = pending_intent
        if pending_options is not None:
            self.pending_options = pending_options



def ask_clarification(goal=None, meal_type=None):
    return {
        "intent": "clarification",
        "question": "Do you want breakfast, lunch, or dinner?",
        "goal": goal,
        "meal_type": meal_type
    }

