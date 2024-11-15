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

        def get_colors(recall_value):
            """Return fill and border colors with different opacities"""
            if recall_value > 0.8:
                return '#006400', '#004b00'  # Dark green
            elif recall_value > 0.6:
                return '#228B22', '#1a6919'  # Forest green
            elif recall_value > 0.4:
                return '#FFA500', '#cc8400'  # Orange
            elif recall_value > 0.2:
                return '#8B0000', '#660000'  # Dark red
            return '#FF0000', '#cc0000'      # Red
        
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
        
        # Add CSS for styling
        css = """
        <style>
            .folium-tooltip {
                background-color: rgba(255, 255, 255, 0.9) !important;
                border: 2px solid rgba(0,0,0,0.2);
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 13px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            
            .folium-popup {
                font-size: 14px;
                font-weight: 500;
            }
            
            .leaflet-interactive {
                transition: fill-opacity 0.2s ease-in-out, 
                          stroke-opacity 0.2s ease-in-out,
                          stroke-width 0.2s ease-in-out;
            }
            
            .leaflet-interactive:hover {
                fill-opacity: 0.9 !important;
                stroke-opacity: 1 !important;
                stroke-width: 3px !important;
            }
            
            @keyframes pulse {
                0% { stroke-width: 2px; }
                50% { stroke-width: 3px; }
                100% { stroke-width: 2px; }
            }
            
            .leaflet-interactive:hover {
                animation: pulse 1.5s infinite;
            }
        </style>
        """
        m.get_root().html.add_child(folium.Element(css))
        
        # Add legend
        legend_html = f'''
            <div style="position: fixed; bottom: 50px; right: 50px; background-color:white;
                 padding:10px; border-radius:5px; border:2px solid grey; z-index:9999;">
                <p style="font-weight: bold; margin-bottom: 8px;">Prediction Correctness</p>
                <p><span style="color:#006400; font-size:16px;">●</span> &gt; 80%</p>
                <p><span style="color:#228B22; font-size:16px;">●</span> 60-80%</p>
                <p><span style="color:#FFA500; font-size:16px;">●</span> 40-60%</p>
                <p><span style="color:#8B0000; font-size:16px;">●</span> 20-40%</p>
                <p><span style="color:#FF0000; font-size:16px;">●</span> &lt; 20%</p>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def create_performance_chart(metrics_df):
    """Create horizontal bar chart for performance metrics"""
    try:
        sorted_data = metrics_df.sort_values(ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sorted_data.index,
            x=sorted_data.values,
            orientation='h',
            text=[f"{x:.1%}" for x in sorted_data.values],
            textposition='auto',
            marker_color=sorted_data.values,
            marker=dict(
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title="Prediction Correctness",
                    tickformat=".0%"
                )
            )
        ))
        
        fig.update_layout(
            xaxis_title="%Prediction Correctness",
            yaxis_title=None,
            height=max(400, len(sorted_data) * 25),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                tickformat=".0%",
                range=[0, max(sorted_data.values) * 1.1]
            ),
            yaxis=dict(automargin=True),
            showlegend=False,
            plot_bgcolor='white',
            bargap=0.2
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=False)
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
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
    
    # Display layout
    col1, col2 = st.columns([5, 3])
    
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
            st_folium(m, width=550, height=650)
    
    with col2:
        filtered_df = df[df[('recall_per_row', 'mean')] >= min_recall]
        if selected_region != 'All':
            filtered_df = filtered_df.loc[[selected_region]]
        
        st.markdown("### %Correctness Stats")
        mean_recall = filtered_df[('recall_per_row', 'mean')].mean()
        median_recall = filtered_df[('recall_per_row', 'mean')].median()
        total_count = filtered_df[('recall_per_row', 'count')].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{mean_recall:.1%}")
        with col2:
            st.metric("Median", f"{median_recall:.1%}")
        with col3:
            st.metric("Address Counts", f"{int(total_count):,}")
        
        st.markdown(f"### By {selected_group}")
        display_df = pd.DataFrame({
            'Mean Recall': filtered_df[('recall_per_row', 'mean')],
        }).round(4)
        
        if not display_df.empty:
            chart = create_performance_chart(display_df['Mean Recall'])
            if chart:
                st.plotly_chart(chart, use_container_width=True)


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


 # streamlit run 2024-11-15_test_app_lat_long_v4.py    