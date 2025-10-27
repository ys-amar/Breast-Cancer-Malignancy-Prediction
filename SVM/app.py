import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Please make sure 'svm_model.pkl' and 'scaler.pkl' are in the same folder as this app.")
        st.stop()

model, scaler = load_model()

# Title
st.title("🎗️ Breast Cancer Diagnosis Predictor")
st.markdown("Enter the 10 mean features to get a prediction.")
st.warning("⚠️ **Disclaimer**: This is for educational purposes only and should not be used for actual medical diagnosis.")

st.markdown("---")

# Create input fields
st.header("📊 Enter Patient Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Group 1")
    radius_mean = st.number_input(
        "Radius Mean", 
        min_value=0.0, 
        max_value=30.0, 
        value=14.0, 
        step=0.1,
        help="Mean of distances from center to points on the perimeter"
    )
    
    texture_mean = st.number_input(
        "Texture Mean", 
        min_value=0.0, 
        max_value=40.0, 
        value=19.0, 
        step=0.1,
        help="Standard deviation of gray-scale values"
    )
    
    perimeter_mean = st.number_input(
        "Perimeter Mean", 
        min_value=0.0, 
        max_value=200.0, 
        value=92.0, 
        step=0.1,
        help="Perimeter of the tumor"
    )
    
    area_mean = st.number_input(
        "Area Mean", 
        min_value=0.0, 
        max_value=2500.0, 
        value=655.0, 
        step=1.0,
        help="Area of the tumor"
    )
    
    smoothness_mean = st.number_input(
        "Smoothness Mean", 
        min_value=0.0, 
        max_value=0.2, 
        value=0.096, 
        step=0.001, 
        format="%.4f",
        help="Local variation in radius lengths"
    )

with col2:
    st.subheader("Feature Group 2")
    compactness_mean = st.number_input(
        "Compactness Mean", 
        min_value=0.0, 
        max_value=0.4, 
        value=0.104, 
        step=0.001, 
        format="%.4f",
        help="Perimeter² / area - 1.0"
    )
    
    concavity_mean = st.number_input(
        "Concavity Mean", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.089, 
        step=0.001, 
        format="%.4f",
        help="Severity of concave portions of the contour"
    )
    
    concave_points_mean = st.number_input(
        "Concave Points Mean", 
        min_value=0.0, 
        max_value=0.3, 
        value=0.048, 
        step=0.001, 
        format="%.4f",
        help="Number of concave portions of the contour"
    )
    
    symmetry_mean = st.number_input(
        "Symmetry Mean", 
        min_value=0.0, 
        max_value=0.4, 
        value=0.181, 
        step=0.001, 
        format="%.4f",
        help="Symmetry of the tumor"
    )
    
    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean", 
        min_value=0.0, 
        max_value=0.1, 
        value=0.063, 
        step=0.001, 
        format="%.4f",
        help="Coastline approximation - 1"
    )

st.markdown("---")

# Predict button
if st.button("🔍 Predict Diagnosis", type="primary", use_container_width=True):
    # Prepare input data (order must match your training data!)
    input_data = np.array([[
        radius_mean, 
        texture_mean, 
        perimeter_mean, 
        area_mean, 
        smoothness_mean, 
        compactness_mean, 
        concavity_mean, 
        concave_points_mean, 
        symmetry_mean, 
        fractal_dimension_mean
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Try to get probability if available
    try:
        probability = model.predict_proba(input_scaled)[0]
        has_probability = True
    except AttributeError:
        has_probability = False
    
    # Display results
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    if has_probability:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("### 🔴 Malignant")
                st.markdown("The tumor is predicted to be **Malignant**")
            else:
                st.success("### 🟢 Benign")
                st.markdown("The tumor is predicted to be **Benign**")
        
        with col2:
            st.metric("Benign Probability", f"{probability[0]*100:.2f}%")
        
        with col3:
            st.metric("Malignant Probability", f"{probability[1]*100:.2f}%")
        
        # Create probability bar chart
        st.markdown("---")
        st.subheader("Probability Distribution")
        
        col_benign, col_malignant = st.columns(2)
        with col_benign:
            st.progress(probability[0], text=f"Benign: {probability[0]*100:.1f}%")
        with col_malignant:
            st.progress(probability[1], text=f"Malignant: {probability[1]*100:.1f}%")
    else:
        # If no probability available (e.g., linear kernel SVM without probability=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            if prediction == 1:
                st.error("### 🔴 Malignant")
                st.markdown("The tumor is predicted to be **Malignant**")
            else:
                st.success("### 🟢 Benign")
                st.markdown("The tumor is predicted to be **Benign**")
        with col2:
            st.info("💡 Probability scores not available for this model configuration")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses your trained SVM model to predict breast cancer diagnosis.
    
    **Features Used:**
    1. Radius Mean
    2. Texture Mean
    3. Perimeter Mean
    4. Area Mean
    5. Smoothness Mean
    6. Compactness Mean
    7. Concavity Mean
    8. Concave Points Mean
    9. Symmetry Mean
    10. Fractal Dimension Mean
    
    **Model:** Support Vector Machine (SVM)
    """)
    
    st.markdown("---")
    st.markdown("**📁 Required Files:**")
    st.code("• svm_model.pkl\n• scaler.pkl")