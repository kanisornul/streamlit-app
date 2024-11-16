import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from pathlib import Path
import json
import branca.colormap as cm
import plotly.graph_objects as go
import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# Run the Streamlit app configuration at the very start
if 'data_loaded' not in st.session_state:
    st.set_page_config(
        page_title="NER Model Performance",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
    )
    st.session_state.data_loaded = True


st.markdown("""
    <style>
        .main > div {padding-top: 0.5rem; padding-left: 1rem; padding-right: 1rem;}
        .stSidebar > div {padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem;}
        [data-testid="stSidebar"] {min-width: 220px !important; max-width: px !important;}
    </style>
""", unsafe_allow_html=True)


# # Modify the data loading and preprocessing function
# @st.cache_data
# def load_and_preprocess_data(file_path):
#     """Load and preprocess data with caching"""
#     try:
#         df = pd.read_csv(file_path)
#         df = df[df['latitude'].notna() & df['longitude'].notna()].drop(columns=[
#             'index_column', 'm_name', 'm_surname', 'm_address_number', 
#             'm_street', 'm_subdistrict', 'm_district', 'm_province', 
#             'm_zipcode', 'name', 'address_number', 'street', 
#             'name_street', 'full_address'
#         ])
#         return df
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return None
# @st.cache_data
# def calculate_metrics(_df, selected_entities):
#     """Calculate metrics based on selected entities"""
#     try:
#         df = _df.copy()
#         if selected_entities:
#             df['recall_per_row'] = df[selected_entities].sum(axis=1) / len(selected_entities)
#         else:
#             df['recall_per_row'] = 0
            
#         # Aggregate by each level
#         aggregated = {}
#         for level in ['province', 'district', 'subdistrict', 'zipcode']:
#             agg = (df.groupby(level)
#                   .agg({
#                       'latitude': 'mean',
#                       'longitude': 'mean',
#                       'recall_per_row': ['mean', 'count']
#                   })
#                   .round(4))
#             aggregated[level] = agg
            
#         return aggregated
#     except Exception as e:
#         st.error(f"Error calculating metrics: {str(e)}")
#         return None

@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and preprocess data with caching"""
    try:
        df = pd.read_csv(file_path)
        df = df[df['latitude'].notna() & df['longitude'].notna()].drop(columns=[
            'index_column', 'm_name', 'm_surname', 'm_address_number', 
            'm_street', 'm_subdistrict', 'm_district', 'm_province', 
            'm_zipcode', 'name', 'address_number', 'street', 
            'name_street', 'full_address'
        ])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def calculate_metrics(_df, selected_entities, selected_data_type='All'):
    """Calculate metrics based on selected entities and data type"""
    try:
        df = _df.copy()
        
        # Filter by data type if specified
        if selected_data_type != 'All':
            df = df[df['data_type'] == selected_data_type]
            
        if selected_entities:
            df['recall_per_row'] = df[selected_entities].sum(axis=1) / len(selected_entities)
        else:
            df['recall_per_row'] = 0
            
        # Aggregate by each level
        aggregated = {}
        for level in ['province', 'district', 'subdistrict', 'zipcode']:
            agg = (df.groupby(level)
                  .agg({
                      'latitude': 'mean',
                      'longitude': 'mean',
                      'recall_per_row': ['mean', 'count']
                  })
                  .round(4))
            aggregated[level] = agg
            
        return aggregated, df  # Return both aggregated data and filtered dataframe
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None, None


# #for province border on hover
# @st.cache_data
# def load_thailand_geojson():
#     """Load Thailand province boundaries"""
#     # You'll need to get Thailand GeoJSON data - this is a placeholder path
#     geojson_path = r"C:\Users\User\OneDrive\Desktop\Chula Stat\Semester 1\Data Visualization\Project 3\thailand-provinces.geojson"
#     with open(geojson_path) as f:
#         return json.load(f)

# Add this helper function
def safe_float_conversion(value):
    """Safely convert Series or float to float value"""
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


def create_optimized_map(df, group_col, selected_level=None, min_recall=0.0, max_points=1000):
    """Create map with enhanced circle markers and proper coordinate handling"""
    thailand_center = [13.7563, 100.5018]
    
    try:
        m = folium.Map(
            location=thailand_center,
            zoom_start=6,
            tiles='CartoDB positron',
            zoom_control=True,
            scrollWheelZoom=True,
            dragging=True,
            min_zoom=5,
            max_zoom=10
        )

        # Create color steps for the legend - reversed order for better visualization
        colors = ['#1E90FF', '#4682B4', '#F0E68C', '#FFA500', '#FF8C00']  # Reversed color order
        colormap = cm.LinearColormap(
            colors=colors,
            vmin=0,
            vmax=100,  # Changed to 100 for percentage
            caption='Prediction Correctness (%)',
            tick_labels=['0%', '20%', '40%', '60%', '80%', '100%']
        )
        
        # Position the colormap vertically on the right
        colormap.add_to(m)
        
        def get_colors(recall_value):
            """Return fill and border colors using a blue-orange palette"""
            if recall_value > 0.8:
                return '#1E90FF', '#1c74d1'  # Dark blue 
            elif recall_value > 0.6:
                return '#4682B4', '#36648B'  # Steel blue 
            elif recall_value > 0.4:
                return '#F0E68C', '#b3a200'  # Yellow
            elif recall_value > 0.2:
                return '#FFA500', '#e57c00'  # Medium orange
            return '#FF8C00', '#e67c00'  # Dark orange

        # Filter data
        filtered_df = df[df[('recall_per_row', 'mean')] >= min_recall]
        if selected_level:
            filtered_df = filtered_df.loc[[selected_level]]
        
        # Sample if too many points
        if len(filtered_df) > max_points:
            filtered_df = filtered_df.sample(n=max_points)
            
        # Add circle markers with enhanced styling
        for idx, row in filtered_df.iterrows():
            lat = safe_float_conversion(row['latitude'])
            lon = safe_float_conversion(row['longitude'])
            recall_value = safe_float_conversion(row[('recall_per_row', 'mean')])
            count = int(row[('recall_per_row', 'count')])
            
            fill_color, border_color = get_colors(recall_value)
            
            info_text = f"{idx}: {recall_value:.1%}"
            detailed_text = f"{idx}: {recall_value:.1%} (n={count})"
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=folium.Popup(detailed_text, max_width=300),
                tooltip=folium.Tooltip(detailed_text),
                color=border_color,
                weight=2,
                fill=True,
                fillColor=fill_color,
                fillOpacity=0.7,
                opacity=0.9,
                name=f'circle_{idx}'
            ).add_to(m)

        # Add custom CSS to position the legend
        css = """
        <style>
        .leaflet-right .legend {
            margin-right: 20px;
        }
        .legend {
            padding: 6px 8px;
            font-size: 12px;
            background: white;
            background: rgba(255,255,255,0.9);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
            line-height: 24px;
            color: #555;
        }
        .caption {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 5px;
            text-align: center;
        }
        </style>
        """
        m.get_root().html.add_child(folium.Element(css))
        
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def main():
    st.title("Model Performance on Prediction Correctness")
    
    # Load data
    raw_df = load_and_preprocess_data(r"correct_dataset2")
    if raw_df is None:
        return
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Data and Filters")
        
        # Data Type filter (moved to top)
        st.markdown("<b>Data Type:</b>", unsafe_allow_html=True)
        data_types = ['All'] + sorted(raw_df['data_type'].unique().tolist())
        selected_data_type = st.selectbox("Select Data Type", data_types)
        
        st.markdown("<hr style='margin: 1rem 0'>", unsafe_allow_html=True)
        
        st.markdown("### %Correctness Calculation")
        
        # Entity Selection section
        st.markdown("<b>Address Components:</b>", unsafe_allow_html=True)
        
        entity_options = {
            'Subdistrict': 'm_subdistrict_flag',
            'District': 'm_district_flag',
            'Province': 'm_province_flag',
            'Postal Code': 'm_zipcode_flag'
        }
        
        # Select All checkbox
        select_all = st.checkbox("Select All", value=True)
        
        # Entity selection
        if select_all:
            selected_entities = list(entity_options.values())
            # Show disabled checkboxes when Select All is true
            for name in entity_options.keys():
                st.checkbox(name, value=True, disabled=True)
        else:
            selected_entities = [
                col for name, col in entity_options.items()
                if st.checkbox(name, value=False)
            ]
        
        # Warning if no entities selected
        if not selected_entities:
            st.warning("Please select at least one component")
        
        # Add separator before other filters
        st.markdown("<hr style='margin: 1rem 0'>", unsafe_allow_html=True)
        
        st.markdown("### Grouping and Filtering")
        group_mapping = {
            'Province': 'province',
            'District': 'district',
            'Subdistrict': 'subdistrict',
            'Postal Code': 'zipcode'
        }
        
        selected_group = st.selectbox("Group By", list(group_mapping.keys()))
        group_col = group_mapping[selected_group]
        
        # Calculate metrics based on selected entities and data type
        aggregated_data, filtered_raw_df = calculate_metrics(raw_df, selected_entities, selected_data_type)
        if aggregated_data is None:
            return
            
        df = aggregated_data[group_col]
        
        # Region filter based on filtered data
        unique_regions = sorted(df.index.unique())
        selected_region = st.selectbox(
            f"Filter by {selected_group}", 
            ['All'] + list(unique_regions)
        )
        
        # Percentage format for min recall
        min_recall = st.slider(
            "Minimum %Prediction Correctness",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            format="%d%%"
        ) / 100.0
        
        max_points = st.slider("Max Points on Map", 100, 1000, 500)
    
# Display layout - modified to remove the bar chart
    col1, col2 = st.columns([7, 4])  # Adjusted ratio to give more space to map
    
    with col1:
        entities_text = ", ".join([k for k, v in entity_options.items() if v in selected_entities])
        st.markdown(f"### By {selected_group}")
        
        # Display data type and entity information
        data_type_text = f"Data Type: {selected_data_type}"
        st.write(data_type_text)
        st.write(f"Example Data: ")
        st.write(f"Address Components for %Correctness Calculation: {entities_text}")
        
        selected_level = None if selected_region == 'All' else selected_region
        m = create_optimized_map(df, group_col, selected_level, min_recall, max_points)
        if m:
            st_folium(m, width=600, height=500)  # Increased size
    
    with col2:
        filtered_df = df[df[('recall_per_row', 'mean')] >= min_recall]
        if selected_region != 'All':
            filtered_df = filtered_df.loc[[selected_region]]
        
        st.markdown("### %Correctness Stats")
        mean_recall = filtered_df[('recall_per_row', 'mean')].mean()
        median_recall = filtered_df[('recall_per_row', 'mean')].median()
        total_count = filtered_df[('recall_per_row', 'count')].sum()
         
        # st.markdown(f"### By {selected_group}")
        # display_df = pd.DataFrame({
        #     'Mean Recall': filtered_df[('recall_per_row', 'mean')],
        # }).round(4)


        # Add CSS styling for reducing font size
        st.markdown(
            """
            <style>
            div[data-testid="metric-container"] {
                font-size: 12px !important; /* Reduce font size */
            }
            div[data-testid="metric-container"] label {
                font-size: 14px !important; /* Optional: Adjust label size separately */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Columns with st.metric components
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{mean_recall:.1%}")
        with col2:
            st.metric("Median", f"{median_recall:.1%}")
        with col3:
            st.metric("Address Counts", f"{int(total_count):,}")



if __name__ == "__main__":
    main()


## Model Part

# Load the pre-trained model
@st.cache_data
def load_model():
    model = joblib.load("NER_model.joblib")
    return model

model = load_model()

# Define stopwords
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5,
    }
    
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True
    
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True
    
    return features

def parse(text):
    tokens = text.split()  # Tokenize the input text by space
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    
    # Make predictions using the model
    prediction = model.predict([features])[0]
    
    return tokens, prediction

# Add explanation mapping for predictions
def map_explanation(label):
    explanation = {
        "LOC": "Location (Tambon, Amphoe, Province)",
        "POST": "Postal Code",
        "ADDR": "Other Address Element",
        "O": "Not an Address"
    }
    return explanation.get(label, "Unknown")

# Set up the Streamlit app
st.title("Let try Named Entity Recognition (NER) model!")

# Example input for NER analysis
example_input = "นายสมชาย เข็มกลัด 254 ถนน พญาไท แขวง วังใหม่ เขต ปทุมวัน กรุงเทพมหานคร 10330"

# Text input for user data with the example as placeholder text
user_text = st.text_area("Enter your address:", value="", placeholder=example_input)

# Button to make predictions
if st.button("Predict!"):
    # Make predictions
    tokens, predictions = parse(user_text)

    # Add explanations to predictions
    explanations = [map_explanation(pred) for pred in predictions]

    # Create a horizontal table
    data = pd.DataFrame([predictions, explanations], columns=tokens, index=["Prediction", "Explanation"])

    # Display the results
    st.write("Tokenized Results and Predictions with Explanations (Horizontal Table):")
    st.dataframe(data)


 # streamlit run 2024-11-15_test_app_lat_long_v5.py    