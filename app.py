# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Customer Churn Prediction App")
st.markdown("""
This app predicts whether a customer is likely to churn based on various features.
Enter the customer information in the form below to get a prediction.
""")

# Load the saved model and related objects
@st.cache_resource
def load_model():
    try:
        with open('churn_model.pkl', 'rb') as file:
            model_dict = pickle.load(file)
        return model_dict
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_dict = load_model()

if model_dict is None:
    st.error("Failed to load the model. Please check the model file.")
    st.stop()

# Extract components from model_dict
try:
    model = model_dict['model']
    selected_features = model_dict['selected_features']
    scaler = model_dict['scaler']
    numeric_cols = model_dict['numeric_cols']
    binary_cols = model_dict['binary_cols']
    st.success("Model loaded successfully!")
except KeyError as e:
    st.error(f"Missing key in model dictionary: {e}")
    st.stop()

# Create the prediction function
def predict_churn(customer_data):
    """Predict churn for a new customer"""
    # Convert to DataFrame
    customer_df = pd.DataFrame([customer_data])
    
    # Scale numeric features
    customer_df[numeric_cols] = scaler.transform(customer_df[numeric_cols])
    
    # Select only the features used in the model
    customer_features = customer_df[selected_features]
    
    # Make prediction
    prediction = model.predict(customer_features)[0]
    probability = model.predict_proba(customer_features)[0][1]
    
    return prediction, probability

# Define the input form
st.header("Customer Information")

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    # Personal information
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    gender_encoded = 0 if gender == "Male" else 1
    
    senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"])
    senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
    
    partner = st.selectbox("Partner", options=["No", "Yes"])
    partner_encoded = 1 if partner == "Yes" else 0
    
    dependents = st.selectbox("Dependents", options=["No", "Yes"])
    dependents_encoded = 1 if dependents == "Yes" else 0
    
    # Service information
    st.subheader("Service Information")
    
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=24)
    
    phone_service = st.selectbox("Phone Service", options=["No", "Yes"])
    phone_service_encoded = 1 if phone_service == "Yes" else 0
    
    multiple_lines_options = ["No", "Yes"] if phone_service == "Yes" else ["No phone service"]
    multiple_lines = st.selectbox("Multiple Lines", options=multiple_lines_options)
    multiple_lines_encoded = 0 if multiple_lines == "No" else (1 if multiple_lines == "Yes" else 2)
    
    internet_service = st.selectbox("Internet Service", options=["No", "DSL", "Fiber optic"])
    internet_service_encoded = 0 if internet_service == "No" else (1 if internet_service == "DSL" else 2)

with col2:
    # Continue service information
    if internet_service != "No":
        online_security = st.selectbox("Online Security", options=["No", "Yes"])
        online_security_encoded = 0 if online_security == "No" else 1
        
        online_backup = st.selectbox("Online Backup", options=["No", "Yes"])
        online_backup_encoded = 0 if online_backup == "No" else 1
        
        device_protection = st.selectbox("Device Protection", options=["No", "Yes"])
        device_protection_encoded = 0 if device_protection == "No" else 1
        
        tech_support = st.selectbox("Tech Support", options=["No", "Yes"])
        tech_support_encoded = 0 if tech_support == "No" else 1
        
        streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes"])
        streaming_tv_encoded = 0 if streaming_tv == "No" else 1
        
        streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes"])
        streaming_movies_encoded = 0 if streaming_movies == "No" else 1
    else:
        online_security_encoded = 2  # No internet service
        online_backup_encoded = 2  # No internet service
        device_protection_encoded = 2  # No internet service
        tech_support_encoded = 2  # No internet service
        streaming_tv_encoded = 2  # No internet service
        streaming_movies_encoded = 2  # No internet service
        
    # Account information
    st.subheader("Account Information")
    
    contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
    contract_encoded = 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)
    
    paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"])
    paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
    
    payment_method = st.selectbox("Payment Method", 
                                 options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    payment_method_encoded = 0 if payment_method == "Electronic check" else (
        1 if payment_method == "Mailed check" else (
            2 if payment_method == "Bank transfer (automatic)" else 3
        )
    )
    
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure, step=0.1)

# Create a customer data dictionary
customer_data = {
    'gender': gender_encoded,
    'SeniorCitizen': senior_citizen_encoded,
    'Partner': partner_encoded,
    'Dependents': dependents_encoded,
    'tenure': tenure,
    'PhoneService': phone_service_encoded,
    'MultipleLines': multiple_lines_encoded,
    'InternetService': internet_service_encoded,
    'OnlineSecurity': online_security_encoded,
    'OnlineBackup': online_backup_encoded,
    'DeviceProtection': device_protection_encoded,
    'TechSupport': tech_support_encoded,
    'StreamingTV': streaming_tv_encoded,
    'StreamingMovies': streaming_movies_encoded,
    'Contract': contract_encoded,
    'PaperlessBilling': paperless_billing_encoded,
    'PaymentMethod': payment_method_encoded,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Prediction button
if st.button("Predict Churn"):
    # Make prediction
    prediction, probability = predict_churn(customer_data)
    
    # Display result
    st.header("Prediction Result")
    
    # Create two columns for the result
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        # Display prediction with color
        if prediction == 1:
            st.error("### Customer is likely to churn")
        else:
            st.success("### Customer is likely to stay")
        
        # Display probability
        st.write(f"Churn probability: {probability:.2f}")
        
        # Risk level
        if probability < 0.3:
            risk = "Low"
            color = "green"
        elif probability < 0.7:
            risk = "Medium"
            color = "orange"
        else:
            risk = "High"
            color = "red"
        
        st.write(f"Risk level: ::{color}[{risk}]")
    
    with result_col2:
        # Create a gauge chart for the probability
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Create the gauge
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.8])
        
        # Create a semi-circle
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax.plot(x, y, color='black')
        
        # Fill color based on probability
        cmap = plt.cm.RdYlGn_r
        theta_p = np.linspace(0, probability * np.pi, 100)
        x_p = r * np.cos(theta_p)
        y_p = r * np.sin(theta_p)
        
        ax.fill_between(x_p, 0, y_p, color=cmap(probability))
        
        # Add needle
        needle_x = r * 0.9 * np.cos(probability * np.pi)
        needle_y = r * 0.9 * np.sin(probability * np.pi)
        ax.plot([0, needle_x], [0, needle_y], color='black', linewidth=2)
        
        # Add circle at needle base
        circle = plt.Circle((0, 0), radius=0.05, color='black')
        ax.add_patch(circle)
        
        # Add labels
        ax.text(-1, -0.2, "0% (Stay)", fontsize=10, ha='left')
        ax.text(1, -0.2, "100% (Churn)", fontsize=10, ha='right')
        ax.text(0, -0.2, "50%", fontsize=10, ha='center')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.3, 1.1)
        
        # Remove axes
        ax.axis('off')
        
        # Display the plot
        st.pyplot(fig)
    
    # Show feature importance if available
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        # Get feature importance
        importances = model.feature_importances_
        features = selected_features
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)
        
        # Show top features
        st.write("### Top Features Influencing Churn")
        for i, row in importance_df.head(5).iterrows():
            st.write(f"- **{row['Feature']}**: {row['Importance']:.4f}")

# Add information about the model
with st.expander("About the Model"):
    st.write("""
    This model uses a Random Forest Classifier to predict customer churn based on selected features.
    
    **Model Performance:**
    - Accuracy: approximately 80%
    - Features used: The top {} features were selected based on their importance.
    
    **How it works:**
    1. Customer data is collected through the form
    2. Data is preprocessed and scaled
    3. The model predicts whether the customer is likely to churn
    4. Results are displayed with probability and risk level
    """.format(len(selected_features)))

# Footer
st.markdown("---")
st.markdown("© 2025 Customer Churn Prediction App | Created with Streamlit")