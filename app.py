import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Title
# -----------------------------
st.title("📩 Spam Message Detection")
st.write("Enter a message to check whether it is Spam or Ham")

# -----------------------------
# Load Dataset (fixed file)
# -----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# -----------------------------
# Train Model
# -----------------------------
X = df['message']
y = df['label_encoded']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("✍️ Enter Your Message")

user_input = st.text_area("Type your message here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚫 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Not Spam)")
