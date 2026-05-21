import streamlit as st
import pandas as pd


from core.chatbot import chatbot, memory
from core.preprocessing import df

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="AI Nutrition Assistant", page_icon="🥗", layout="wide")

st.title("🥗 AI Nutrition Assistant")

st.markdown("""
Smart AI-powered nutrition system:

- 🍽️ Meal recommendation  
- 🔍 Meal analysis  
- 🔄 Substitution  
- 📊 Comparison  
- 📅 Daily & weekly plans  
- 🧠 Memory-based chat  
""")


st.markdown(
    """
<style>



.stColumn  {
    width: fit-content !important;
    flex: none !important;
}

div[data-testid="stButton"]{
    display:flex;
    justify-content:center;
}

div[data-testid="stButton"] > button {
    border: none !important;
    background: transparent !important;

    color: #1f77b4 !important;

    text-decoration: underline dotted !important;
    text-underline-offset: 4px;

    box-shadow: none !important;

    padding: 0 !important;
    margin: 0 !important;

    min-height: auto !important;
    height: auto !important;

    font-size: 15px !important;
    font-weight: 500 !important;

    width: auto !important;
}

div[data-testid="stButton"] > button:hover {
    background: transparent !important;
    color: #0d5aa7 !important;
}

div[data-testid="stButton"] > button:focus {
    outline: none !important;
    box-shadow: none !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Session State Init
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_clarification(response, msg_index):
    st.info(response["question"])

    options = response.get("options", [])

    if not options:
        st.warning(
            "I did not found the meal name in my database. Please try again with a different name or ask for a substitute."
        )
        return

    cols = st.columns(len(options))

    for i, option in enumerate(options):
        with cols[i]:
            unique_key = f"clar_{msg_index}_{i}"

            if st.button(option, key=unique_key):
                st.session_state.chat_history.append(
                    {"role": "user", "content": option}
                )

                reply = chatbot(option)

                if reply is None:
                    reply = {
                        "intent": "unknown",
                        "type": "text",
                        "results": "No response returned.",
                    }

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply}
                )

                st.rerun()


def render_comparison(comp):
    if isinstance(comp, dict) and "error" in comp:
        st.error(comp["error"])
        return

    meal_a = comp["meal_a"]
    meal_b = comp["meal_b"]

    winner = comp.get("winner_for_goal")
    goal = comp.get("goal", "maintenance")

    st.subheader("📊 Meal Comparison")

    # =====================================================
    # Winner
    # =====================================================
    if winner:
        st.success(f"🏆 Best choice for {goal.title()}: {winner}")

    st.info(f"Why? {', '.join(comp['winner_reason'])}")

    # =====================================================
    # Meal Cards
    # =====================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"## 🍽️ {meal_a['Name']}")

        st.metric("Calories", f"{meal_a['Calories']} kcal")

        st.metric("Protein", f"{meal_a['ProteinContent']} g")

        st.metric("Carbs", f"{meal_a['CarbohydrateContent']} g")

        st.metric("Fat", f"{meal_a['FatContent']} g")

    with col2:
        st.markdown(f"## 🍽️ {meal_b['Name']}")

        st.metric("Calories", f"{meal_b['Calories']} kcal")

        st.metric("Protein", f"{meal_b['ProteinContent']} g")

        st.metric("Carbs", f"{meal_b['CarbohydrateContent']} g")

        st.metric("Fat", f"{meal_b['FatContent']} g")

    # =====================================================
    # Table Comparison
    # =====================================================
    st.markdown("### 📋 Nutrition Comparison")

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Calories",
                "Protein",
                "Carbohydrates",
                "Fat",
            ],
            meal_a["Name"]: [
                meal_a["Calories"],
                meal_a["ProteinContent"],
                meal_a["CarbohydrateContent"],
                meal_a["FatContent"],
            ],
            meal_b["Name"]: [
                meal_b["Calories"],
                meal_b["ProteinContent"],
                meal_b["CarbohydrateContent"],
                meal_b["FatContent"],
            ],
        }
    )

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # =====================================================
    # Smart Summary
    # =====================================================
    st.markdown("### 🧠 AI Summary")

    summary = []

    if meal_a["ProteinContent"] > meal_b["ProteinContent"]:
        summary.append(f"✅ {meal_a['Name']} has higher protein.")
    else:
        summary.append(f"✅ {meal_b['Name']} has higher protein.")

    if meal_a["Calories"] > meal_b["Calories"]:
        summary.append(f"🔥 {meal_a['Name']} is higher in calories.")
    else:
        summary.append(f"🔥 {meal_b['Name']} is higher in calories.")

    if meal_a["FatContent"] < meal_b["FatContent"]:
        summary.append(f"🥗 {meal_a['Name']} is lower in fat.")
    else:
        summary.append(f"🥗 {meal_b['Name']} is lower in fat.")

    for item in summary:
        st.write(item)


def render_response(response: dict, idx):
    if response is None:
        st.warning("No response returned from chatbot.")
        return

    if not isinstance(response, dict):
        st.write(response)
        return

    intent = response.get("intent", "unknown")
    print(f"Rendering response for intent: {intent}")

    if intent == "greeting" and response.get("results") is not None:
        # st.info(response["results"])
        st.write(response["results"])

    elif response.get("type") == "clarification":
        render_clarification(response, idx)
        return

    if intent == "recommend_meal":
        st.subheader("🍽️ Recommended Meal")
        st.dataframe(response["results"])

    elif intent == "list_meals":
        st.subheader("📋 Meal List")
        st.dataframe(response["results"])

    elif intent == "pre_workout":
        st.subheader("🏋️‍♂️ Pre-Workout Meal")
        st.dataframe(response["results"])

    elif intent == "post_workout":
        st.subheader("💪 Post-Workout Meal")
        st.dataframe(response["results"])

    elif intent == "daily_plan":
        st.subheader("📅 Daily Plan")
        for meal_type, meal_df in response["plan"].items():
            st.markdown(f"### {meal_type.capitalize()}")
            st.dataframe(meal_df)

    elif intent == "weekly_plan":
        st.subheader("📆 Weekly Plan")
        st.dataframe(response["plan"])

    elif intent == "analyze_meal":
        print(f"Rendering analysis response: {response}")
        st.subheader("🔍 Meal Analysis")
        st.write(response["results"])

    elif intent == "substitute_meal":
        st.subheader("🔄 Substitutes")
        st.write(response["results"])

    elif intent == "compare_meals":

        comp = response["results"]

        render_comparison(comp)


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🧠 Memory & History")

st.sidebar.subheader("🧠 Stored Memory")

memory_data = {
    "Goal": memory.last_goal,
    "Meal Type": memory.last_meal_type,
    "Meal Name": memory.last_meal_name,
    "Compare Meals": memory.last_compare_meals,
    "Pending Intent": memory.pending_intent,
}

has_memory = False


for key, value in memory_data.items():
    if value not in (None, "", [], {}):
        has_memory = True
        st.sidebar.write(f"**{key}:** {value}")


if not has_memory:
    st.sidebar.info("No memory stored")


# ----------------------------
# عرض الأسئلة السابقة
# ----------------------------
# st.sidebar.subheader("🕘 Previous Questions")

# display only the last 10 questions for brevity
# if st.session_state.chat_history:
    # for i, item in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
    #     role = item["role"].capitalize()
    #     text = item["content"]
    #     st.sidebar.markdown(f"{i}. {role}:** {text}")
#     pass
# else:
#     st.sidebar.info("No previous questions")


if st.sidebar.button("🗑️ Clear Memory"):
    memory.clear()
    st.session_state.chat_history = []
    st.rerun()


# ----------------------------
# Show all chat history like ChatGPT
# ----------------------------
for idx, item in enumerate(st.session_state.chat_history):
    with st.chat_message(item["role"]):
        if item["role"] == "assistant":
            render_response(item["content"], idx)
        else:
            st.write(item["content"])


# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Ask me anything about nutrition...")

if user_input:
    # خزّن الرسالة الجديدة فقط للعرض داخل الجلسة
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # استدعاء chatbot الأساسي
    response = chatbot(user_input)

    print(f"Chatbot response: {response}")

    # خزّن رد المساعد للعرض فقط، وليس كذاكرة دائمة
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.rerun()
