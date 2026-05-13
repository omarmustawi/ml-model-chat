from core.recommendation import recommend_meals
import pandas as pd


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
