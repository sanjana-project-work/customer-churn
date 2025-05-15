import streamlit as st
import pandas as pd
import numpy as np
st.title("Customer Churn Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Label Encoding and One-Hot Encoding
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        if df[column].nunique() == 2:
            df[column] = le.fit_transform(df[column])
        else:
            df = pd.get_dummies(df, columns=[column])

    df = df.dropna()
    df.columns = df.columns.str.strip()

    if 'Exited' not in df.columns:
        st.error("❌ 'Exited' column not found.")
    else:
        # Features and labels
        drop_cols = [col for col in ['Exited', 'RowNumber', 'CustomerId'] if col in df.columns]
        X = df.drop(drop_cols, axis=1)
        y = df['Exited']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model training
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Metrics
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Not Churned', 'Churned'], columns=['Predicted Not Churned', 'Predicted Churned'])
        st.dataframe(cm_df)

        # Feature Importance
        st.subheader("Top 15 Feature Importances")
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(15)
        st.bar_chart(importances.set_index('Feature'))

        # KDE and line plots using Plotly
        if 'Age' in df.columns:
            st.subheader("Age Distribution by Churn")
            age_dist = df[['Age', 'Exited']]
            fig = px.histogram(age_dist, x='Age', color='Exited', barmode='overlay', nbins=40)
            st.plotly_chart(fig)

            st.subheader("Average Churn Rate by Age")
            age_churn = df.groupby('Age')['Exited'].mean().reset_index()
            st.line_chart(age_churn.set_index('Age'))

            st.subheader("Boxplot of Age by Churn")
            fig_box = px.box(df, x='Exited', y='Age', points="all", labels={"Exited": "Churned"})
            st.plotly_chart(fig_box)

        # Optional: Pair plot with selected numeric features
        numeric_columns = ['Age', 'CreditScore', 'Balance', 'Tenure', 'EstimatedSalary']
        available_columns = [col for col in numeric_columns if col in df.columns]
        if all(col in df.columns for col in available_columns):
            st.subheader("Scatter Matrix of Numeric Features")
            scatter_df = df[available_columns + ['Exited']]
            fig = px.scatter_matrix(scatter_df, dimensions=available_columns, color='Exited')
            st.plotly_chart(fig)
