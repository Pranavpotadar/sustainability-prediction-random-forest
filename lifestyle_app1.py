import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Eco-Life Rating",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .rating-box {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .category-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Define categories (same as before)
categories = {
    'Location': ['Urban', 'Suburban', 'Rural'],
    'DietType': ['Mostly Plant-Based', 'Balanced', 'Mostly Animal-Based'],
    'LocalFoodFrequency': ['Often', 'Sometimes', 'Rarely', 'Always'],
    'TransportationMode': ['Bike', 'Public Transit', 'Car', 'Walk'],
    'EnergySource': ['Renewable', 'Mixed', 'Non-Renewable'],
    'HomeType': ['Apartment', 'House', 'Other'],
    'ClothingFrequency': ['Rarely', 'Sometimes', 'Often', 'Always'],
    'CommunityInvolvement': ['High', 'Moderate', 'Low'],
    'Gender': ['Female', 'Male', 'Non-Binary', 'Prefer not to say'],
    'UsingPlasticProducts': ['Rarely', 'Sometimes', 'Often', 'Never'],
    'DisposalMethods': ['Composting', 'Recycling', 'Landfill', 'Combination'],
    'PhysicalActivities': ['High', 'Moderate', 'Low']
}

@st.cache_resource
def train_model():
    """Train a new model using sample data"""
    # Create sample data similar to your training data
    sample_data = []
    np.random.seed(42)
    
    for _ in range(500):  # Create 500 sample records
        sample = {
            'Age': np.random.randint(18, 80),
            'Location': np.random.randint(0, 3),
            'DietType': np.random.randint(0, 3),
            'LocalFoodFrequency': np.random.randint(0, 4),
            'TransportationMode': np.random.randint(0, 4),
            'EnergySource': np.random.randint(0, 3),
            'HomeType': np.random.randint(0, 3),
            'HomeSize': np.random.randint(500, 5000),
            'ClothingFrequency': np.random.randint(0, 4),
            'SustainableBrands': np.random.choice([True, False]),
            'EnvironmentalAwareness': np.random.randint(1, 6),
            'CommunityInvolvement': np.random.randint(0, 3),
            'MonthlyElectricityConsumption': np.random.randint(50, 500),
            'MonthlyWaterConsumption': np.random.randint(1000, 5000),
            'Gender': np.random.randint(0, 4),
            'UsingPlasticProducts': np.random.randint(0, 4),
            'DisposalMethods': np.random.randint(0, 4),
            'PhysicalActivities': np.random.randint(0, 3)
        }
        
        # Calculate rating based on sustainable choices
        sustainable_points = 0
        if sample['TransportationMode'] in [0, 1, 3]:  # Bike, Public Transit, Walk
            sustainable_points += 1
        if sample['EnergySource'] == 0:  # Renewable
            sustainable_points += 1
        if sample['SustainableBrands']:
            sustainable_points += 1
        if sample['EnvironmentalAwareness'] >= 4:
            sustainable_points += 1
        if sample['UsingPlasticProducts'] <= 1:  # Rarely or Never
            sustainable_points += 1
        
        # Assign rating based on points
        sample['Rating'] = min(max(sustainable_points + 1, 1), 5)
        sample_data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(sample_data)
    
    # Split features and target
    X = df.drop('Rating', axis=1)
    y = df['Rating']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def create_gauge_chart(value, title):
    """Create a gauge chart for the sustainability score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 5], 'tickwidth': 1},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 1], 'color': 'lightgray'},
                {'range': [1, 2], 'color': 'lightpink'},
                {'range': [2, 3], 'color': 'lightyellow'},
                {'range': [3, 4], 'color': 'lightgreen'},
                {'range': [4, 5], 'color': 'palegreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def create_radar_chart(input_data):
    """Create a radar chart for sustainability factors"""
    categories = ['Transport', 'Energy', 'Waste', 'Consumption', 'Community']
    
    # Calculate scores for each category (0-100)
    transport_score = 100 if input_data['TransportationMode'] in [0, 1, 3] else 50
    energy_score = 100 if input_data['EnergySource'] == 0 else 50
    waste_score = 100 if input_data['DisposalMethods'] in [0, 1] else 50
    consumption_score = 100 if input_data['UsingPlasticProducts'] in [0, 3] else 50
    community_score = 100 if input_data['CommunityInvolvement'] == 0 else 50
    
    values = [transport_score, energy_score, waste_score, consumption_score, community_score]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400
    )
    return fig

def main():
    # Sidebar for user info and tips
    with st.sidebar:
        st.image("https://media-content.angi.com/2c74c46f-2738-4408-9f0b-163cee10bb0b.jpg", width=100)
        st.title("🌱 Eco-Life Rating")
        st.markdown("---")
        st.markdown(""" ### Tips for a Sustainable Life
        1. 🚶‍♂️ Walk or bike when possible
        2. ♻️ Reduce, Reuse, Recycle
        3. 🌿 Choose plant-based meals
        4. 💡 Use energy-efficient appliances
        5. 🛍️ Support sustainable brands
        """)
        st.markdown("---")
        st.markdown("### About")
        st.info("This app helps you evaluate your lifestyle's sustainability and provides personalized recommendations for improvement.")

    # Main content
    st.title("🌍 Sustainable Lifestyle Assessment")
    st.write("Evaluate your environmental impact and get personalized recommendations.")

    # Progress bar for sections
    progress = 0
    progress_bar = st.progress(progress)
    
    # Initialize session state for multi-step form
    if 'step' not in st.session_state:
        st.session_state.step = 1

    # Multi-step form
    if st.session_state.step == 1:
        st.markdown("### 🏠 Living Situation")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.age = st.number_input('Age', min_value=18, max_value=100, value=30)
                st.session_state.location = st.selectbox('Location', categories['Location'])
                st.session_state.home_type = st.selectbox('Home Type', categories['HomeType'])
            with col2:
                st.session_state.home_size = st.number_input('Home Size (sq ft)', min_value=100, max_value=10000, value=1000)
                st.session_state.energy = st.selectbox('Energy Source', categories['EnergySource'])
                st.session_state.gender = st.selectbox('Gender', categories['Gender'])
        
        if st.button("Next →"):
            st.session_state.step = 2
            progress = 33
            progress_bar.progress(progress)

    elif st.session_state.step == 2:
        st.markdown("### 🍃 Lifestyle Choices")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.diet_type = st.selectbox('Diet Type', categories['DietType'])
                st.session_state.local_food = st.selectbox('Local Food Frequency', categories['LocalFoodFrequency'])
                st.session_state.transport = st.selectbox('Transportation Mode', categories['TransportationMode'])
            with col2:
                st.session_state.clothing_freq = st.selectbox('Shopping Frequency', categories['ClothingFrequency'])
                st.session_state.sustainable_brands = st.checkbox('Do you prefer sustainable brands?')
                st.session_state.physical = st.selectbox('Physical Activities Level', categories['PhysicalActivities'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 1
                progress = 0
        with col2:
            if st.button("Next →"):
                st.session_state.step = 3
                progress = 66
                progress_bar.progress(progress)

    elif st.session_state.step == 3:
        st.markdown("### 🌿 Environmental Impact")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.electricity = st.number_input('Monthly Electricity Consumption (kWh)', min_value=0, max_value=1000, value=250)
                st.session_state.water = st.number_input('Monthly Water Consumption (gallons)', min_value=0, max_value=10000, value=3000)
                st.session_state.plastic = st.selectbox('Plastic Product Usage', categories['UsingPlasticProducts'])
            with col2:
                st.session_state.disposal = st.selectbox('Disposal Methods', categories['DisposalMethods'])
                st.session_state.env_awareness = st.slider('Environmental Awareness (1-5)', 1, 5, 3)
                st.session_state.community = st.selectbox('Community Involvement', categories['CommunityInvolvement'])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 2
                progress = 33
        with col2:
            if st.button("Calculate Rating"):
                progress = 100
                progress_bar.progress(progress)
                
                # Create input data dictionary and make prediction
                try:
                    input_data = {
                        'Age': st.session_state.age,
                        'Location': categories['Location'].index(st.session_state.location),
                        'DietType': categories['DietType'].index(st.session_state.diet_type),
                        'LocalFoodFrequency': categories['LocalFoodFrequency'].index(st.session_state.local_food),
                        'TransportationMode': categories['TransportationMode'].index(st.session_state.transport),
                        'EnergySource': categories['EnergySource'].index(st.session_state.energy),
                        'HomeType': categories['HomeType'].index(st.session_state.home_type),
                        'HomeSize': st.session_state.home_size,
                        'ClothingFrequency': categories['ClothingFrequency'].index(st.session_state.clothing_freq),
                        'SustainableBrands': st.session_state.sustainable_brands,
                        'EnvironmentalAwareness': st.session_state.env_awareness,
                        'CommunityInvolvement': categories['CommunityInvolvement'].index(st.session_state.community),
                        'MonthlyElectricityConsumption': st.session_state.electricity,
                        'MonthlyWaterConsumption': st.session_state.water,
                        'Gender': categories['Gender'].index(st.session_state.gender),
                        'UsingPlasticProducts': categories['UsingPlasticProducts'].index(st.session_state.plastic),
                        'DisposalMethods': categories['DisposalMethods'].index(st.session_state.disposal),
                        'PhysicalActivities': categories['PhysicalActivities'].index(st.session_state.physical)
                    }

                    # Get model prediction
                    model, scaler = train_model()
                    input_df = pd.DataFrame([input_data])
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)[0]

                    # Display results
                    st.markdown("---")
                    st.markdown("## Your Sustainability Results")
                    
                    # Display gauge chart
                    st.plotly_chart(create_gauge_chart(prediction, "Sustainability Score"), use_container_width=True)
                    
                    # Display radar chart
                    st.markdown("### Sustainability Breakdown")
                    st.plotly_chart(create_radar_chart(input_data), use_container_width=True)
                    
                    # Interpretations and recommendations
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 🎯 Rating Interpretation")
                        interpretations = {
                            1: "Your lifestyle has significant room for improvement in terms of sustainability.",
                            2: "You're making some sustainable choices, but there's potential for more environmental consciousness.",
                            3: "You maintain a moderately sustainable lifestyle with a balanced approach.",
                            4: "Great job! You're making many sustainable choices in your daily life.",
                            5: "Excellent! You're living a highly sustainable lifestyle and making a positive impact!"
                        }
                        st.info(interpretations[prediction])
                    
                    with col2:
                        st.markdown("### 🌱 Personalized Recommendations")
                        recommendations = []
                        if not st.session_state.sustainable_brands:
                            recommendations.append("🛍️ Consider switching to sustainable brands")
                        if st.session_state.transport == 'Car':
                            recommendations.append("🚲 Try using public transit or biking")
                        if st.session_state.plastic in ['Often', 'Sometimes']:
                            recommendations.append("♻️ Reduce plastic usage")
                        if st.session_state.energy == 'Non-Renewable':
                            recommendations.append("💡 Look into renewable energy options")
                        if st.session_state.disposal == 'Landfill':
                            recommendations.append("🗑️ Start recycling and composting")
                        
                        for rec in recommendations:
                            st.write(rec)
                    
                    # Reset button
                    if st.button("Start Over"):
                        st.session_state.step = 1
                        st.experimental_rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.write("Please check your inputs and try again.")

if __name__ == '__main__':
    main()
