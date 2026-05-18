from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChatMemory:
    last_goal: Optional[str] = None
    last_meal_type: Optional[str] = None
    last_meal_name: Optional[str] = None
    last_compare_meals: Optional[list[str]] = None

    pending_intent: Optional[str] = None
    pending_options: Optional[list[str]] = None

    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    # def add_message(self, role: str, text: Any) -> None:
    #     self.conversation_history.append(
    #         {
    #             "role": role,
    #             "text": text,
    #         }
    #     )
    #     self.conversation_history = self.conversation_history[-20:]

    _MISSING = object()

    def update(
        self,
        *,
        last_goal: Optional[str] = None,
        last_meal_type: Optional[str] = None,
        last_meal_name: Optional[str] = None,
        last_compare_meals: Optional[list[str]] = None,
        pending_intent: Optional[str] = None,
        pending_options: Optional[list[str]] = None,
    ) -> None:
        if last_goal is not self._MISSING:
            self.last_goal = last_goal

        if last_meal_type is not self._MISSING:
            self.last_meal_type = last_meal_type

        if last_meal_name is not self._MISSING:
            self.last_meal_name = last_meal_name

        if last_compare_meals is not self._MISSING:
            self.last_compare_meals = last_compare_meals

        if pending_intent is not self._MISSING:
            self.pending_intent = pending_intent

        if pending_options is not self._MISSING:
            self.pending_options = pending_options

    def clear(self) -> None:
        self.last_goal = None
        self.last_meal_type = None
        self.last_meal_name = None
        self.last_compare_meals = None
        self.pending_intent = None
        self.pending_options = None
        self.conversation_history.clear()


def ask_clarification(
    intent=None, 
    goal=None, 
    meal_type=None, 
    meal_name=None, 
    compare_meals=None, 
    options = None,
):
    if intent == "analyze_meal":
        if not meal_name:
            return {
                "intent": intent,
                "type": "clarification",
                "slot": "meal_name",
                "question": "What is the name of the meal you want to know about?",
                "options": options or [],
                "goal": goal,
                "meal_type": meal_type,
                "meal_name": meal_name,
                "compare_meals": compare_meals,
                "missing": ["meal_name"],
            }
        
        
        
    if intent == "substitute_meal":
        if not meal_name:
            return {
                "intent": intent,
                "type": "clarification",
                "slot": "meal_name",
                "question": "What is the name of the meal you want to find a substitute for?",
                "options": options or [],
                "goal": goal,
                "meal_type": meal_type,
                "meal_name": meal_name,
                "compare_meals": compare_meals,
                "missing": ["meal_name"],
            }
        

    if intent == "compare_meals":
        if not compare_meals or len(compare_meals) < 2:
            return {
                "intent": intent,
                "type": "clarification",
                "slot": "compare_meals",
                "question": "Which meals do you want to compare? Please provide at least two meal names.",
                "options": options or [],
                "goal": goal,
                "meal_type": meal_type,
                "meal_name": meal_name,
                "compare_meals": compare_meals,
                "missing": ["compare_meals"],
            }
        
        
        


    if not goal:
        return {
            "intent": intent,
            "type": "clarification",
            "slot": "goal",
            "question": "What is your nutrition goal? ",
            "options": ["muscle gain", "fat loss", "maintenance"],
            "goal": goal,
            "meal_type": meal_type,
            "meal_name": meal_name,
            "compare_meals": compare_meals,
            "missing": ["goal"],
        }

    if not meal_type:
        return {
            "intent": intent,
            "type": "clarification",
            "slot": "meal_type",
            "question": "What type of meal do you want? ",
            "options": ["breakfast", "lunch", "dinner", "snack"],
            "goal": goal,
            "meal_type": meal_type,
            "missing": ["meal_type"],
        }

    if not meal_name:
        return {
            "intent": intent,
            "type": "clarification",
            "slot": "meal_name",
            "question": "What is the name of the meal you want to know about?",
            "options": [],
            "goal": goal,
            "meal_type": meal_type,
            "meal_name": meal_name,
            "missing": ["meal_name"],
        }

    return {
        "intent": intent,
        "type": "clarification",
        "slot": None,
        "question": "Need clarification",
        "options": [],
        "goal": goal,
        "meal_type": meal_type,
        "meal_name": meal_name,
        "compare_meals": compare_meals,
    }

    # missing = []

    # if not goal:
    #     missing.append("goal")

    # if not meal_type:
    #     missing.append("meal type")

    # if len(missing) == 2:
    #     question = (
    #         "What is your goal? "
    #         "(muscle gain, fat loss, maintenance) "
    #         "and what type of meal do you want?"
    #     )
    # elif "goal" in missing:
    #     question = (
    #         "What is your nutrition goal? "
    #         "(muscle gain, fat loss, maintenance)"
    #     )
    # else:
    #     question = (
    #         "What type of meal do you want? "
    #         "(breakfast, lunch, dinner, snack)"
    #     )

    # return {
    #     "intent": intent,
    #     "type": "clarification",
    #     "question": question,
    #     "goal": goal,
    #     "meal_type": meal_type,
    #     "missing": missing,
    # }
