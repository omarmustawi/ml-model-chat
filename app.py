import streamlit as st
import pandas as pd

from core.chatbot import chatbot
from core.preprocessing import df


# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(
    page_title="AI Nutrition Assistant",
    page_icon="🥗",
    layout="wide"
)

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

# ----------------------------
# Sidebar (filters فقط UI)
# ----------------------------
st.sidebar.title("⚙️ Settings")

selected_goal = st.sidebar.selectbox(
    "Select Goal",
    [None, "muscle gain", "fat loss", "maintenance", "endurance"]
)

selected_meal_type = st.sidebar.selectbox(
    "Select Meal Type",
    [None, "breakfast", "lunch", "dinner", "snack", "dessert"]
)


# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Ask me anything about nutrition...")

if user_input:

    # عرض رسالة المستخدم
    st.chat_message("user").write(user_input)

    # استدعاء chatbot الأساسي
    response = chatbot(user_input)
    print(f"response: {response}")
    

    # ----------------------------
    # عرض الرد
    # ----------------------------
    with st.chat_message("assistant"):

        intent = response.get("intent")

        # greeting
        if "greeting" in response:
            st.info(response["greeting"])

        # ----------------------------
        # RECOMMENDATION
        # ----------------------------
        if intent == "recommend_meal":
            st.subheader("🍽️ Recommended Meal")

            st.dataframe(response["results"])

        # ----------------------------
        # ANALYSIS
        # ----------------------------
        elif intent == "analyze_meal":
            st.subheader("🔍 Meal Analysis")

            st.write(response["analysis"])

        # ----------------------------
        # SUBSTITUTE
        # ----------------------------
        elif intent == "substitute_meal":
            st.subheader("🔄 Substitutes")

            st.write(response["substitutes"])

        # ----------------------------
        # COMPARE
        # ----------------------------
        elif intent == "compare_meals":
            st.subheader("📊 Comparison Result")

            comp = response["comparison"]

            if "error" in comp:
                st.error(comp["error"])
            else:
                st.json(comp)

        # ----------------------------
        # DAILY PLAN
        # ----------------------------
        elif intent == "daily_plan":
            st.subheader("📅 Daily Plan")

            plan = response["plan"]

            for meal_type, meal_df in plan.items():
                st.markdown(f"### {meal_type.capitalize()}")
                st.dataframe(meal_df)

        # ----------------------------
        # WEEKLY PLAN
        # ----------------------------
        elif intent == "weekly_plan":
            st.subheader("📆 Weekly Plan")

            st.dataframe(response["plan"])

        # ----------------------------
        # UNKNOWN / ERROR
        # ----------------------------
        elif intent == "unknown":
            st.warning(response["message"])

        elif intent == "clarification":
            st.info(response.get("question", "Need clarification"))
            if "options" in response:
                st.write("Options:", response["options"])

        else:
            st.write(response)