# 🚀 Classification - Flipkart Customer Service Satisfaction
### Machine Learning & GenAI for Proactive Support Optimization

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Sklearn](https://img.shields.io/badge/Library-ScikitLearn-orange.svg)
![GenAI](https://img.shields.io/badge/GenAI-HuggingFace-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

---

## 📖 Project Summary
This project is an end-to-end Data Science and Generative AI solution designed to optimize Customer Satisfaction (CSAT) for **Flipkart**. By analyzing historical customer support data, we developed a system that not only predicts customer satisfaction scores but also automatically understands the root causes of dissatisfaction.

The project solves two core business problems: **Reactive Support** (finding out a customer is unhappy only after they leave) and **Unstructured Feedback Analysis** (the inability to manually read thousands of complaints).

We achieved this by:
1.  Building a **Classification Model (Random Forest)** to predict CSAT scores (1-5) based on interaction metrics.
2.  Handling real-world data challenges like **Class Imbalance** using **SMOTE**.
3.  Integrating a **Generative AI Pipeline (DistilBART)** to automatically summarize negative customer remarks.

---

## 🏢 Business Context & Problem Statement
**The Challenge:**
Customer support teams often operate reactively. Analyzing thousands of tickets manually to find the root cause of dissatisfaction is inefficient and unscalable. Furthermore, standard predictive models struggle to identify at-risk customers due to the high prevalence of positive ratings (class imbalance).

**The Goal:**
To build a proactive Machine Learning system that predicts CSAT scores based on interaction metadata (e.g., handling time, agent tenure) and deploys a GenAI solution to instantly categorize unstructured customer complaints for strategic decision-making.

---

## ⚙️ Technical Architecture
The project follows a structured data science lifecycle:

1.  **Data Preprocessing:**
    * Handled missing values (Imputed Price with Median, Agent Shift with Mode).
    * Converted timestamps to calculate 'Response Time'.
    * Removed outliers in 'Connected Handling Time' (capped at 99th percentile).
2.  **Exploratory Data Analysis (EDA):**
    * Identified severe **Class Imbalance** (~80% of ratings were 5-star).
    * Discovered that **Handling Time** had a strong negative correlation with CSAT.
3.  **Feature Engineering:**
    * Created `Response_Time_Min` (Time taken to respond).
    * One-Hot Encoding for categorical variables (Channel, Category).
4.  **Model Implementation:**
    * **Algorithm:** Random Forest Classifier.
    * **Imbalance Handling:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples for minority classes (Scores 1 & 2).
5.  **GenAI Integration:**
    * Utilized **Hugging Face Transformers (DistilBART)** for text summarization.
    * Implemented a "Demo Mode" to showcase capabilities using complex manual examples (due to data sparsity in the raw dataset).

---

## 📊 Key Insights
1.  **Primary Driver of Dissatisfaction:** The Random Forest feature importance analysis revealed that **Connected Handling Time** is the #1 predictor of CSAT. Long call durations significantly drop satisfaction scores.
2.  **Operational Efficiency:** Customers prefer quick resolutions over "senior" agents. Agent Tenure had less impact than speed of resolution.
3.  **Data Quality Gap:** A major finding was that dissatisfied customers often leave the "Remarks" field blank. This highlights a business need to enforce minimum character limits on feedback forms to enable better AI analysis.

---

## 🤖 GenAI Showcase
We implemented a summarization pipeline using `sshleifer/distilbart-cnn-12-6`.

**Example Output:**
> **Original Complaint:** *"I waited on hold for 45 minutes and when the agent finally answered, they were extremely rude and didn't know how to process my refund. I am very disappointed with the service."*
>
> **AI Summary:** *"Agent was rude and did not know how to process refund."*

---

## 🛠️ Libraries Used
* **Pandas & NumPy:** Data Manipulation.
* **Matplotlib & Seaborn:** Data Visualization.
* **Scikit-Learn:** Machine Learning (Random Forest) & Evaluation.
* **Imbalanced-learn:** SMOTE for class balancing.
* **Transformers (Hugging Face):** Generative AI pipeline.

---

## 📈 Future Scope
* **Azure Deployment:** Deploy the model as a real-time API endpoint on **Microsoft Azure App Service**.
* **Live Dashboard:** Integrate the model predictions into a PowerBI dashboard for live agent monitoring.
* **Advanced NLP:** Upgrade to **Azure OpenAI (GPT-4)** for sentiment analysis and automated email drafting.
