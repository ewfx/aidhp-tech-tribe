import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast  # For safely evaluating strings into dictionaries

# Step 1: Define Wells Fargo Credit Card Bonuses
credit_cards = {
    "Basic Card": {
        "Restaurants": 0.02, "Shopping": 0.02, "Entertainment": 0.02,
        "Benefits": "No annual fee, 0% APR for the first 15 months, unlimited 2% cashback on all purchases."
    },
    "Standard Card": {
        "Restaurants": 0.03, "Shopping": 0.01, "Entertainment": 0.03,
        "Benefits": "3x points on dining, travel, and streaming services, no foreign transaction fees."
    },
    "Reflect Card": {
        "Restaurants": 0.01, "Shopping": 0.05, "Entertainment": 0.01,
        "Benefits": "5% cashback on shopping for the first $2,500 spent each quarter, extended 0% APR for 18 months."
    },
    "WRewards Card": {
        "Restaurants": 0.02, "Shopping": 0.03, "Entertainment": 0.01,
        "Benefits": "Earn 20,000 bonus points after spending $1,000 in the first 3 months, redeemable for travel or cash."
    },
    "Propel Card": {
        "Restaurants": 0.03, "Shopping": 0.02, "Entertainment": 0.04,
        "Benefits": "4x points on entertainment, 3x points on dining and travel, no annual fee."
    },
    "Platinum Card": {
        "Restaurants": 0.01, "Shopping": 0.01, "Entertainment": 0.01,
        "Benefits": "Basic card with no annual fee, ideal for building credit, includes cell phone protection."
    },
    "Signature Card": {
        "Restaurants": 0.04, "Shopping": 0.02, "Entertainment": 0.03,
        "Benefits": "4% cashback on dining, 3% on entertainment, complimentary concierge service."
    },
    "Cash Wise Card": {
        "Restaurants": 0.015, "Shopping": 0.015, "Entertainment": 0.015,
        "Benefits": "1.5% unlimited cashback on all purchases, $150 sign-up bonus after spending $500 in the first 3 months."
    },
    "Student Card": {
        "Restaurants": 0.02, "Shopping": 0.02, "Entertainment": 0.02,
        "Benefits": "No annual fee, designed for students, includes tools to help build credit responsibly."
    },
    "Business Card": {
        "Restaurants": 0.03, "Shopping": 0.03, "Entertainment": 0.02,
        "Benefits": "3% cashback on business-related expenses, free employee cards, and expense tracking tools."
    },

    "Travel Rewards Card": {
        "Restaurants": 0.02, "Shopping": 0.01, "Entertainment": 0.03,
        "Benefits": "Earn 5x points on travel bookings, complimentary airport lounge access, no foreign transaction fees."
    },
    "Green Card": {
        "Restaurants": 0.01, "Shopping": 0.02, "Entertainment": 0.01,
        "Benefits": "2% cashback on eco-friendly purchases, carbon offset contributions for every dollar spent."
    },
    "Luxury Card": {
        "Restaurants": 0.05, "Shopping": 0.03, "Entertainment": 0.04,
        "Benefits": "5% cashback on fine dining, complimentary travel insurance, and exclusive event access."
    },
    "Cashback Plus Card": {
        "Restaurants": 0.02, "Shopping": 0.04, "Entertainment": 0.02,
        "Benefits": "4% cashback on online shopping, 2% on all other purchases, no annual fee."
    }
}

# Step 2: Load the Trained Model and Synthetic Data
@st.cache_data
def load_model_and_data():
    # Load the trained model
    model = joblib.load("../model/trained_model.joblib")
    print("Loaded trained model.")

    # Load the synthetic data (used for encoding consistency)
    customer_data = pd.read_csv("../dataset/customer_dataset.csv")
    print("Loaded synthetic data.")

    # Convert Monthly_Spending from string to dictionary
    customer_data["Monthly_Spending"] = customer_data["Monthly_Spending"].apply(ast.literal_eval)

    return model, customer_data


# Step 3: Calculate Total Spending per Category
def get_total_spending_per_category(customer_spending):
    """
    Calculate the total spending for each category over the last 3 months.
    """
    total_spending_per_category = {}
    for month in customer_spending.values():
        for category, items in month.items():
            if category not in total_spending_per_category:
                total_spending_per_category[category] = 0
            total_spending_per_category[category] += sum(items.values())
    return total_spending_per_category


# Step 4: Evaluate Monthly Expenses with Each Card
def evaluate_monthly_expenses(total_spending_per_category, credit_cards):
    """
    Calculate the total monthly expenses after applying credit card bonuses.
    """
    card_savings = {}
    for card_name, bonuses in credit_cards.items():
        total_expenses = 0
        for category, spending in total_spending_per_category.items():
            if category in bonuses:
                total_expenses += spending * (1 - bonuses[category])  # Apply discount
            else:
                total_expenses += spending  # No discount
        card_savings[card_name] = total_expenses
    return card_savings


# Step 5: Generate Recommendations
def generate_recommendations(customer_data, customer_name):
    # Retrieve the customer's information
    customer_info = customer_data[customer_data["Name"] == customer_name]
    if customer_info.empty:
        return f"Customer '{customer_name}' not found in the dataset."

    customer_info = customer_info.iloc[0].to_dict()

    # Get total spending per category
    total_spending_per_category = get_total_spending_per_category(customer_info["Monthly_Spending"])

    # Evaluate monthly expenses with each card
    card_savings = evaluate_monthly_expenses(total_spending_per_category, credit_cards)

    # Identify the best card
    best_card = min(card_savings, key=card_savings.get)
    savings = sum(total_spending_per_category.values()) - card_savings[best_card]

    # Prepare the recommendation message
    recommendation = (
        f"The best credit card for minimizing expenses is the **{best_card}**.\n"
        f"Using this card, you can save ${savings:.2f} per month.\n"
        f"Total Monthly Expenses with {best_card}: ${card_savings[best_card]:.2f}"
    )
    return recommendation


# Step 6: Streamlit Chatbot Interface
def main():
    st.title("💳 Personalized Credit Card Recommendation Tool")
    st.write(
        "Welcome! I'm here to help you find the best Wells Fargo credit card to minimize your expenses based on your spending habits.")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load the model and data
    model, customer_data = load_model_and_data()

    # User input
    user_input = st.chat_input("Enter your name:")
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        response = generate_recommendations(customer_data, user_input)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# Run the app
if __name__ == "__main__":
    main()