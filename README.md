# AI-Driven-Energy-Expenditure-Prediction-System
An interactive, machine learning–powered application that predicts calorie burn during exercise using biometric and physiological inputs. Built using a Random Forest Regression model and deployed through an advanced Streamlit dashboard with real-time analytics and dynamic UI components.

# 📌 Overview

This project estimates the number of calories burned during physical activity based on biometric and exercise-related data such as heart rate, body temperature, age, and duration. The system uses a machine learning model trained on 200k+ data samples and provides intelligent fitness insights like MET values, intensity levels, and BMI classification.

The goal is to offer a personalized, data-driven fitness analytics tool that performs real-time inference in an intuitive and visually rich web interface.

# 🚀 Features

🔮 AI-Powered Calorie Prediction using a Random Forest Regressor.

🧬 Biometric Input Panel for personalized physiological data.

⚡ Real-Time Inference using optimized preprocessing pipelines.

# 📊 Advanced Fitness Metrics:

MET (Metabolic Equivalent)

BMI & weight classification

Exercise intensity level

Calories per minute

# ✨ Modern UI/UX with animations, neon cards, glassmorphism, and interactive analytics.

📈 Insights Dashboard with dynamic visuals powered by Plotly.

🌐 Deployed using Streamlit for seamless web interaction.

# 🧠 Machine Learning Pipeline

Label Encoding for categorical feature ("Sex")

Min-Max Scaling for numerical normalization

Consistent feature ordering to match training schema

Random Forest Regression for calorie prediction

Hyperparameter tuning and model evaluation (94% accuracy)

Joblib model serialization for deployment

# 🏗️ Project Structure

├── app.py                 # Streamlit application
├── final_model.joblib     # Trained Random Forest model
├── energy_burn_moderator.ipynb  # Training & experimentation notebook
├── requirements.txt        # Dependencies
└── README.md               # Documentation

# 🛠️ Technologies Used

Python 3.x

Scikit-learn — Random Forest Regressor, preprocessing tools

Pandas & NumPy — data processing

Streamlit — interactive UI and deployment

Plotly — visual analytics & graphs

Joblib — model persistence

Matplotlib/Seaborn (optional, for EDA)

# 📐 How It Works

User enters biometric data (sex, weight, height, age).

User enters exercise metrics (heart rate, body temperature, duration).

Inputs pass through the preprocessing pipeline (encoding + scaling).

Random Forest model predicts calories burned.

Dashboard displays insights (MET, intensity, BMI, calories/min, etc.).

# 👥 Developed by Vaishnav M - www.linkedin.com/in/vaishnav-m-

# 🙏 Acknowledgments

Special thanks to:

Mentor: Dixon Joy - https://www.linkedin.com/in/dixson-joy-527513173/

Institution: Luminar Technolab, Kochi under Rahul Mohanakumar @ https://www.linkedin.com/company/luminartechnolab/posts/?feedView=all @ https://www.linkedin.com/in/rahulluminar/
