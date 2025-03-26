# 🚀 Hyper-Personalized Marketing for Financial Products

## 📌 Table of Contents
- [🎯 Introduction](#-introduction)
- [🎥 Demo](#-demo)
- [💡 Inspiration](#-inspiration)
- [⚙️ What It Does](#-what-it-does)
- [🛠️ How We Built It](#-how-we-built-it)
- [🚧 Challenges We Faced](#-challenges-we-faced)
- [🏃 How to Run](#-how-to-run)
- [🏗️ Tech Stack](#-tech-stack)
- [👥 Team](#-team)

---

## 🎯 Introduction
Financial customers are more likely to buy products when they feel an emotional connection with the brand. This project uses **Generative AI** and **Machine Learning** to create personalized marketing campaigns based on customer preferences, spending habits, and financial goals. Instead of generic offers, our model generates tailored recommendations, improving engagement and conversion rates.

---

## 🎥 Demo
🔗 **Live Demo**: In the artifacts directory

---

## 💡 Inspiration
Marketing reports show that **generic promotions fail to engage modern customers**. People expect **personalized recommendations** that match their lifestyle. Traditional marketing struggles to offer this level of customization at scale.  
This project leverages **AI-driven automation** to generate **hyper-personalized** financial offers, eliminating the need for manual campaign creation.

---

## ⚙️ What It Does
✅ **Analyzes Customer Behavior**: Uses financial data to understand spending habits.  
✅ **Generates Personalized Offers**: AI-driven recommendations based on individual financial profiles.  
✅ **Automates Marketing Campaigns**: Removes manual effort in designing customer-specific promotions.  
✅ **Enhances Customer Engagement**: Creates emotional connections with financial brands.

---

## 🛠️ How We Built It
- **Data Preprocessing**: Feature engineering on customer financial data.
- **Machine Learning Model**: **CatBoostClassifier** for predictive analysis.
- **Oversampling**: **SMOTE** to handle class imbalance.
- **Retrieval-Augmented Generation (RAG)**: Enhances recommendations.
- **Web Interface**: Built using **Streamlit** for easy interaction.

---

## 🚧 Challenges We Faced
⚡ **Data Imbalance**: Solved using **SMOTE** to improve model training.  
⚡ **Feature Selection**: Required extensive tuning to choose the best financial indicators.  
⚡ **Real-Time Recommendation Accuracy**: Optimized model hyperparameters for better results.

---

## 🏃 How to Run

1️⃣ Clone the repository

https://github.com/ewfx/aidhp-tech-tribe.git

2️⃣ Install dependencies

pip install -r requirements.txt
3️⃣ Run the project
python train.py
streamlit run app.py

🏗️ Tech Stack
🔹 Frontend: Streamlit
🔹 Backend: FastAPI
🔹 Machine Learning: CatBoost, Scikit-learn
🔹 Database: Elasticsearch
🔹 Other: OpenAI API, Hugging Face

👥 Team
👤 Kashish Yusuf
👤 Sankeerth Sankar