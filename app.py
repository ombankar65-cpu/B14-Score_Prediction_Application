import streamlit as st
import pickle
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for shadow and card styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #2b6cb0;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: none;
    }
    .stButton>button:hover {
        background-color: #2c5282;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open('knn.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🎓 Student Performance Predictor")
st.write("Enter the student details below to generate a prediction using your deployed KNN model.")

with st.form("prediction_form"):
    st.subheader("Student Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course", ["Engineering", "Science", "Arts", "Commerce", "Business"])
        study_hours = st.slider("Study Hours (per day)", 0.0, 15.0, 4.0, 0.5)
        class_attendance = st.slider("Class Attendance (%)", 0, 100, 75)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        
    with col2:
        sleep_hours = st.slider("Sleep Hours (per night)", 0.0, 12.0, 7.0, 0.5)
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"])
        study_method = st.selectbox("Study Method", ["Self-Study", "Online Videos", "Group Study", "Coaching"])
        facility_rating = st.slider("Facility Rating (1-10)", 1, 10, 7)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])

    submit_button = st.form_submit_button(label="Predict Score")

if submit_button:
    # Encoding categorical inputs to match standard label encoding order.
    # Adjust these mappings if your training label encoder used a different order.
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    internet_map = {"Yes": 1, "No": 0}
    
    course_map = {"Engineering": 0, "Science": 1, "Arts": 2, "Commerce": 3, "Business": 4}
    sleep_quality_map = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
    study_method_map = {"Self-Study": 0, "Online Videos": 1, "Group Study": 2, "Coaching": 3}
    exam_difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}

    input_data = pd.DataFrame([[
        age, 
        gender_map.get(gender, 0), 
        course_map.get(course, 0), 
        study_hours, 
        class_attendance, 
        internet_map.get(internet_access, 0), 
        sleep_hours, 
        sleep_quality_map.get(sleep_quality, 0), 
        study_method_map.get(study_method, 0), 
        facility_rating, 
        exam_difficulty_map.get(exam_difficulty, 0)
    ]], columns=[
        'age', 'gender', 'course', 'study_hours', 'class_attendance', 
        'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
        'facility_rating', 'exam_difficulty'
    ])
    
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Result: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
