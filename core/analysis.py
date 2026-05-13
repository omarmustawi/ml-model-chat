import pandas as pd
import difflib

from sklearn.metrics.pairwise import cosine_similarity


from core.preprocessing import df

from core.recommendation import nutrition_score, profile_vectorizer, profile_matrix 
from utils.text_utils import normalize_text


def format_analysis_colloquial(row):
    return (
        f"Sure, the meal '{row['Name']}' is a {row['Meal Type']} suitable for {row['Goal']}.\n"
        f"It is suitable for the goal of {row['Goal']} because it has {row['Calories']} calories, "
        f"{row['ProteinContent']}g of protein, {row['CarbohydrateContent']}g of carbohydrates, and "
        f"{row['FatContent']}g of fat."
        f"In short: It is a good meal if you want a clear, balanced choice with basic ingredients: {row['Ingredients']}."
    )


def format_substitutes_colloquial(meal_name, substitutes_df):
    if substitutes_df is None or len(substitutes_df) == 0:
        return f"Sorry, I couldn't find any substitutes for '{meal_name}'."

    lines = [f"Here are some healthier substitutes for '{meal_name}':"]
    for i, (_, row) in enumerate(substitutes_df.iterrows(), start=1):
        lines.append(
            f"{i}) **{row['Name']}** — {row['Calories']} calories "
            f"protein: {row['ProteinContent']}g, carbohydrates: {row['CarbohydrateContent']}g, fat: {row['FatContent']}g. "
            f"[type: {row['Meal Type']}]"
        )

    lines.append(
        "In short:These are good options if you want something similar but in a different order or a more relaxed approach."
    )

    return "\n".join(lines)


def analyze_meal(meal_name, style="structured"):
    row = get_meal_row(meal_name)
    if row is None:
        return {"error": f"Meal not found: {meal_name}"}
    result = {
        "Name": row["Name"],
        "Goal": row["Goal"],
        "Meal Type": row["Meal Type"],
        # "Cuisine": row["Cuisine"],
        "Calories": int(row["Calories"]),
        "ProteinContent": int(row["ProteinContent"]),
        "CarbohydrateContent": int(row["CarbohydrateContent"]),
        "FatContent": int(row["FatContent"]),
        "Ingredients": row["Ingredients"],
    }

    if style == "colloquial":
        return format_analysis_colloquial(row)

    return result


def substitute_meal(meal_name, top_n=5, style="structured"):
    row = get_meal_row(meal_name)
    if row is None:
        error_df = pd.DataFrame([{"error": f"Meal not found: {meal_name}"}])
        return (
            f"I couldn't find the meal '{meal_name}'. Please check the name and try again."
            if style == "colloquial"
            else error_df
        )

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

    cols = [
        "Name",
        "Goal",
        "Meal Type",
        "Calories",
        "ProteinContent",
        "CarbohydrateContent",
        "FatContent",
        "final_score",
    ]
    result = (
        frame.sort_values("final_score", ascending=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )

    if style == "colloquial":
        return format_substitutes_colloquial(meal_name, result)
    return result


def compare_meals(meal_a, meal_b, goal=None):
    # print(f"Comparing '{meal_a}' vs '{meal_b}' for goal: {goal}")
    row_a = get_meal_row(meal_a)
    row_b = get_meal_row(meal_b)
    # print("Row A:", row_a["Name"] if row_a is not None else "Not found")
    # print("Row B:", row_b["Name"] if row_b is not None else "Not found")

    if row_a is None or row_b is None:
        return {"error": "One or both meals were not found."}

    goal = goal or "maintenance"

    df_pair = pd.DataFrame([row_a, row_b]).copy()
    df_pair["nutrition_score"] = nutrition_score(df_pair, goal=goal)

    winner_idx = df_pair["nutrition_score"].idxmax()
    winner = df_pair.loc[winner_idx, "Name"]

    return {
        "meal_a": row_a[
            ["Name", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]
        ].to_dict(),
        "meal_b": row_b[
            ["Name", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent"]
        ].to_dict(),
        "goal": goal,
        "winner_for_goal": winner,
    }


def get_meal_row(meal_name):
    match = df[df["Name"].str.lower() == normalize_text(meal_name)]
    if len(match) > 0:
        return match.iloc[0]
    # fallback fuzzy
    candidates = difflib.get_close_matches(
        meal_name, df["Name"].tolist(), n=1, cutoff=0.55
    )
    if candidates:
        return df[df["Name"] == candidates[0]].iloc[0]
    return None
