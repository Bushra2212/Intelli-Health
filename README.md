# 🩺 IntelliHealth – Intelligent Health Monitoring System

IntelliHealth is a **machine learning–based web application** designed to monitor and analyze a user’s **stress levels**, **sleep quality**, and **calorie expenditure**.  
The system provides **personalized health insights**, **visual analytics**, and **user-wise health history** through an interactive web interface.

---

## 🌐 Live Application
👉 https://intellihealth-bushra.streamlit.app  
*(Replace with your actual Streamlit URL if different)*

---

## 🎯 Project Objectives
- To analyze daily health parameters using machine learning models
- To predict:
  - 🧠 Stress level  
  - 😴 Sleep quality  
  - 🔥 Calorie expenditure
- To provide **personalized recommendations**
- To maintain **user-specific health history**
- To visualize health indicators for better decision-making

---

## 🧩 Features
- 🔐 **User Authentication (Login & Signup)**
- 👤 **Session-based access control**
- 🧠 **Stress Analysis using ML**
- 😴 **Sleep Quality Prediction**
- 🔥 **Calorie Burn Estimation**
- 📊 **Visualization Dashboard**
- 📁 **User-wise Health History**
- ✅ **Personalized Health Recommendations**
- ☁️ **Deployed as a live web application**

---

## 🛠️ Technologies Used
- **Frontend / Web Framework:** Streamlit  
- **Programming Language:** Python  
- **Machine Learning:** scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Model Storage:** Joblib  
- **Version Control:** Git & GitHub  
- **Large File Handling:** Git LFS  
- **Deployment:** Streamlit Community Cloud  

---

## 🧠 Machine Learning Models
The system uses pre-trained ML models for prediction:
- `stress_model.pkl`
- `sleep_model.pkl`
- `calorie_model.pkl`

Feature lists for each model are stored separately to ensure correct input mapping.

---

## 📂 Project Structure
Intelli-Health/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── stress_model.pkl
├── sleep_model.pkl
├── calorie_model.pkl
├── stress_features.pkl
├── sleep_features.pkl
├── calorie_features.pkl
├── .gitignore
├── .gitattributes
└── README.md


> ⚠️ User data files (`users.csv`, `health_history.csv`) are generated dynamically at runtime and are not committed to GitHub.

---

## 🚀 Deployment
The application is deployed using **Streamlit Community Cloud** and integrated directly with GitHub.

Steps:
1. Push project to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy using `app.py` as the main file

---

## 🎓 Academic Relevance
- Suitable as a **B.Tech CSE (AIML) Major Project**
- Demonstrates:
  - Machine learning integration
  - Decision support systems
  - Web-based deployment
  - User-centric system design

---

## 📌 Future Enhancements
- Password hashing for improved security
- Database integration (SQLite / Firebase)
- User health trend analytics
- Doctor/Admin dashboard
- Mobile-friendly UI enhancements

---

## 👩‍💻 Developed By
**Bushra Fathima (Team Lead)**  
**Sambar Nikitha**  
**Atyam Jayita**  

B.Tech CSE (AIML)  
Institute of Aeronautical Engineering, Hyderabad


---

## 📜 License
This project is developed for **academic and educational purposes**.
