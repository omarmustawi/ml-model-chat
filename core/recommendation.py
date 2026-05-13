from core.preprocessing import df

from sklearn.metrics.pairwise import cosine_similarity
from core.helpers import extract_goal, extract_meal_type, extract_workout_intent



from sklearn.feature_extraction.text import TfidfVectorizer

from core.preprocessing import df



profile_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1
)

profile_matrix = profile_vectorizer.fit_transform(
    df["meal_profile"]
)


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
    cols = ["Name", "Goal", "Meal Type", "Calories", "ProteinContent", "CarbohydrateContent", "FatContent", "final_score"]
    return result[cols].reset_index(drop=True)



def recommend_with_feedback(query, bandit, top_n=5):
    recs = recommend_meals(query, top_n=top_n)
    if recs.empty:
        return recs
    chosen = bandit.choose(recs["Name"].tolist())
    return recs, chosen
