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
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from wordcloud import WordCloud, get_single_color_func
import os
from st_flexible_callout_elements import flexible_error, flexible_success, flexible_warning, flexible_info

# theme
primary_color = "#1e90ff"
default_text_color ="#555555"

#just_added
# At the top of your script, add this new session state variable
if 'first_load' not in st.session_state:
    st.session_state.first_load = True


# Run the Streamlit app configuration at the very start
if 'data_loaded' not in st.session_state:
    st.set_page_config(
        page_title="NER Model Performance",
        page_icon='🤖',
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
    )

    # Initialize filter settings when app first loads
    st.session_state.filter_settings = {
        'data_type': 'All',
        'selected_entities': ['m_subdistrict_flag', 'm_district_flag', 'm_province_flag', 'm_zipcode_flag'],
        'selected_group': 'Province',
        'selected_region': 'All',
        'min_recall': 0.0,
        'group_col': 'province'
    }
    st.session_state.data_loaded = True
    # Add show_popup state
    st.session_state.show_popup = True
    st.session_state.needs_initial_apply = True  # New flag


st.markdown("""
    <style>
        /* Your existing styles */
        .main > div {padding-top: 0.1rem; padding-left: 1rem; padding-right: 1rem;}
        .stSidebar > div {padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem;}
        [data-testid="stSidebar"] {min-width: 220px !important; max-width: px !important;}
        
        /* Updated tooltip styles */
        .header-with-icon {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .tooltip-icon {
            position: relative;
            display: inline-block;
            cursor: help;
            padding-top: 5px;  /* Align with header */
        }
        .tooltip-text {
            display: none;
            position: absolute;
            width: 250px;  /* Adjusted width */
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9em;
            line-height: 1.4;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 100;
            left: -125px;  /* Center the tooltip */
            top: 25px;     /* Position below the icon */
        }
        .tooltip-icon:hover .tooltip-text {
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

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

# Add this helper function
def safe_float_conversion(value):
    """Safely convert Series or float to float value"""
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


# Define the color dictionary with thresholds
color_dict = {
    'very_high_recall': {
        'color': ('#1E90FF', '#1c74d1'),  # Dark blue
        'threshold': 0.8
    },
    'high_recall': {
        'color': ('#4682B4', '#36648B'),  # Steel blue
        'threshold': 0.6
    },
    'medium_recall': {
        'color': ('#F0E68C', '#b3a200'),  # Yellow
        'threshold': 0.4
    },
    'low_recall': {
        'color': ('#FFA500', '#e57c00'),  # Medium orange
        'threshold': 0.2
    },
    'very_low_recall': {
        'color': ('#FF8C00', '#e67c00'),  # Dark orange
        'threshold': 0.0
    }
}

@st.cache_data
def create_optimized_map(df, group_col, selected_level=None, min_recall=0.0, max_recall=1.0): #, max_points=1000):
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

        # Create discrete color steps with branca
        colormap = cm.StepColormap(
            colors=['#FF8C00', '#FFA500', '#F0E68C', '#4682B4', '#1E90FF'],
            vmin=0,
            vmax=100,
            index=[0, 20, 40, 60, 80, 100],
            caption='Prediction Correctness (%)',
            text_color=default_text_color
        )

        colormap.add_to(m)
        

        # Function to get colors based on recall value
        def get_colors_for_map(recall_value):
            """Return color based on recall value using thresholds in the color_dict"""
            for recall_category, data in color_dict.items():
                if recall_value > data['threshold']:
                    return data['color']  # Return the fill color
            return color_dict['very_low_recall']['color'] # Default return if no category is found

        # Filter data
        # filtered_df = df[df[('recall_per_row', 'mean')] >= min_recall]
        filtered_df = df[
                            (df[('recall_per_row', 'mean')] >= min_recall) & 
                            (df[('recall_per_row', 'mean')] <= max_recall)
                        ]
        if selected_level:
            filtered_df = filtered_df.loc[[selected_level]]
        

        # Add circle markers with enhanced styling
        for idx, row in filtered_df.iterrows():
            lat = safe_float_conversion(row['latitude'])
            lon = safe_float_conversion(row['longitude'])
            recall_value = safe_float_conversion(row[('recall_per_row', 'mean')])
            count = int(row[('recall_per_row', 'count')])
            
            fill_color, border_color = get_colors_for_map(recall_value)
            
            info_text = f"{idx}: {recall_value:.1%}"
            detailed_text = f"{idx}: {recall_value:.1%} (Address sample={count})"
            
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

# Create word cloud

font_path_ttf = 'THSarabunNew.ttf'


def get_colors_for_word(recall_value):
    """Return fill color based on recall value"""
    for category, data in color_dict.items():
        if recall_value > data['threshold']:
            return data['color'][0]  # Return only the first color (fill color)
    return color_dict['very_low_recall']['color'][0]  # Default to the first color for very low recall

# Create a custom color function for the word cloud
class CustomColorFunc(object):
    """Create a color function object which assigns colors based on recall value"""
    def __init__(self, df):
        self.df = df

    def __call__(self, word, font_size, position, orientation, random_state=None, **kwargs):
        recall_value = self.df.loc[word, 'mean']  # Get the recall value for the word
        fill_color= get_colors_for_word(recall_value)  # Get colors
        return fill_color  # You can adjust this to return border_color if needed


@st.cache_data
def create_word_cloud_top(df, selected_group='Province'):
    """Create word clouds for top regions with error handling"""
    try:
        # Check if dataframe is empty or has less than required data
        if df.empty:
            st.warning("No data available for word clouds.")
            return None, None #, None # Changed to return only two values

        # Convert tuple index to string if it's a tuple
        if isinstance(df.index[0], tuple):
            df.index = [' '.join(map(str, idx)) for idx in df.index]

        # Always convert index to string (for postal codes)
        df.index = df.index.astype(str)

        # Create a simpler DataFrame with just the required columns
        simple_df = pd.DataFrame({
            'mean': df[('recall_per_row', 'mean')],
            'count': df[('recall_per_row', 'count')]
        })

        # Sort values
        sorted_df = simple_df.sort_values('mean', ascending=False)

        # Get top and bottom 10 (or less if fewer entries available)
        n_items = min(10, len(sorted_df))
        if n_items == 0:
            st.warning("Not enough data points for word clouds.")
            return None, None #, None

        top_10 = sorted_df.head(n_items)
        # bottom_10 = sorted_df.tail(n_items)

        # Create word cloud data - scale values to make them more visible
        top_dict = {
            str(idx): max(float(value) * 100, 1.0)  # Ensure minimum value of 1
            for idx, value in top_10['mean'].items()
        }

        # Create word clouds
        wc_top = WordCloud(
            width=600, height=300,
            background_color='white',
            min_font_size=10,
            max_font_size=100,
            font_path=font_path_ttf,
            # color_func=SimpleColorFunc('blue'),
            color_func=CustomColorFunc(top_10),
            prefer_horizontal=0.7
        ).generate_from_frequencies(top_dict)

        # Create figure with subplots
        fig, (ax1) = plt.subplots(1, 1, figsize=(20, 4))
        fig.patch.set_facecolor('white')

        # Plot top regions
        ax1.imshow(wc_top, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title(f'Most Accurate {selected_group}s', pad=20, fontsize=16, color=default_text_color)


        # Adjust layout
        plt.tight_layout(pad=3)

        # Set font for matplotlib
        plt.rcParams['font.family'] = 'THSarabunNew'

        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)

        return buf, top_10 #, bottom_10

    except Exception as e:
        st.error(f"Error creating word clouds: {str(e)}")
        return None, None #, None

@st.cache_data
def create_word_cloud_bottom(df, selected_group='Province'):
    """Create word clouds for bottom regions with error handling"""
    try:
        # Check if dataframe is empty or has less than required data
        if df.empty:
            st.warning("No data available for word clouds.")
            return None, None #, None # Changed to return only two values

        # Convert tuple index to string if it's a tuple
        if isinstance(df.index[0], tuple):
            df.index = [' '.join(map(str, idx)) for idx in df.index]

        # Always convert index to string (for postal codes)
        df.index = df.index.astype(str)

        # Create a simpler DataFrame with just the required columns
        simple_df = pd.DataFrame({
            'mean': df[('recall_per_row', 'mean')],
            'count': df[('recall_per_row', 'count')]
        })

        # Sort values
        sorted_df = simple_df.sort_values('mean', ascending=True)

        # Get top and bottom 10 (or less if fewer entries available)
        n_items = min(10, len(sorted_df))
        if n_items == 0:
            st.warning("Not enough data points for word clouds.")
            return None, None 

        # adjusted to head after ascedning = True
        bottom_10 = sorted_df.head(n_items)


        bottom_dict = {
            str(idx): max(float(value) * 100, 1.0)  # Ensure minimum value of 1
            for idx, value in bottom_10['mean'].items()
        }

        # Configure word cloud settings
        class SimpleColorFunc(object):
            """Create a color function object which assigns DIFFERENT SHADES of
               specified color(s) to words.
            """
            def __init__(self, color):
                self.color = color

            def __call__(self, word, font_size, position, orientation, random_state=None, **kwargs):
                return self.color

        wc_bottom = WordCloud(
            width=600, height=300,
            background_color='white',
            min_font_size=10,
            max_font_size=100,
            font_path=font_path_ttf,
            color_func=CustomColorFunc(bottom_10),
            prefer_horizontal=0.7
        ).generate_from_frequencies(bottom_dict)

        # Create figure with subplots
        fig, (ax1) = plt.subplots(1, 1, figsize=(20, 4))
        fig.patch.set_facecolor('white')

        # Plot top regions
        ax1.imshow(wc_bottom, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title(f'Least Accurate {selected_group}s', pad=20, fontsize=16, color=default_text_color)


        # Adjust layout
        plt.tight_layout(pad=3)

        # Set font for matplotlib
        plt.rcParams['font.family'] = 'THSarabunNew'

        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)

        return buf, bottom_10 

    except Exception as e:
        st.error(f"Error creating word clouds: {str(e)}")
        return None, None 

def main():
    st.title("Model Performance on Prediction Correctness")

    # Model description
    flexible_info("Welcome! This app lets you explore how a simple CRF-based Named Entity Recognition (NER) model predicts Thai address components. It tags each part as LOC (tambon, amphoe, or province), POST (postal code), ADDR (other address elements), or O (non-address elements).")

    # Load data
    raw_df = load_and_preprocess_data(r"correct_dataset3")
    if raw_df is None:
        return
    
    # Sidebar filters
    with st.sidebar:

        #just_added
        # Show popup at the top of sidebar if not dismissed
        if st.session_state.get('show_popup', True):
            st.warning("Click 'Apply Filter' to update your settings.")
            if st.button("Got it"):
                st.session_state.show_popup = False
                st.rerun()


        # st.markdown("### Data and Filters")
        # st.markdown("### Address Format")
        st.markdown("""
            <div class="header-with-icon">
                <h3 style="margin: 0;">Address Format</h3>
                <div class="tooltip-icon">
                    ℹ️
                    <div class="tooltip-text">
                        Multiple Thai address formats were generated to evaluate variations in model performance based on %Prediction Correctness.
                        <br><br>
                        The dataset combines real Thai administrative data (district, subdistrict, zipcode, and province) with synthetic data (names, house numbers, and streets) created using the Faker library.
                        <br><br>
                        Three address formats were tested:
                        <br>
                        1. ตำบล อำเภอ: Generated addresses include "ตำบล" and "อำเภอ" before subdistrict and district.
                        <br>
                        2. ต. อ.: Generated addresses include "ต." and "อ." before subdistrict and district.
                        <br>
                        3. No Prefix: No prefixes are added before any address components.
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Data Type filter
        # st.markdown("<b>Data Type:</b>", unsafe_allow_html=True)
        data_types = ['All'] + sorted(raw_df['data_type'].unique().tolist())

        # Change description
        temp_data_type_list = [
            "ตำบล อำเภอ" if data_type == "Correct with Full Prefix" else
            "ต. อ." if data_type == "Correct with Short Prefix" else
            "No Prefix" if data_type == "Correct without Prefix" else
            "All"
            for data_type in data_types
]
        temp_data_type = st.selectbox("Select Address Format", options=data_types, format_func=lambda x: temp_data_type_list[data_types.index(x)])
                


        st.markdown("<hr style='margin: 1rem 0'>", unsafe_allow_html=True)
        st.markdown("""
            <div class="header-with-icon">
                <h3 style="margin: 0;">%Correctness Calculation</h3>
                <div class="tooltip-icon">
                    ℹ️
                    <div class="tooltip-text">
                        Prediction Correctness (%) measures how accurately the NER model identifies predefined Thai address components (district, subdistrict, zipcode, and province) in auto-generated data.
                        <br><br>
                        It is calculated as the number of correctly predicted address components divided by the total number of selected address components, expressed as a percentage.
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )


        # Entity Selection section
        st.markdown("<b>Address Components:</b>", unsafe_allow_html=True)
        
        entity_options = {
            'Subdistrict': 'm_subdistrict_flag',
            'District': 'm_district_flag',
            'Province': 'm_province_flag',
            'Postal Code': 'm_zipcode_flag'
        }
        
        # Select All checkbox
        temp_select_all = st.checkbox("Select All", value=True)
        
        # Entity selection
        if temp_select_all:
            temp_selected_entities = list(entity_options.values())
            # Show disabled checkboxes when Select All is true
            for name in entity_options.keys():
                st.checkbox(name, value=True, disabled=True)
        else:
            temp_selected_entities = [
                col for name, col in entity_options.items()
                if st.checkbox(name, value=False)
            ]
        
        # Warning if no entities selected
        if not temp_selected_entities:
            st.warning("Please select at least one component")
        
        st.markdown("<hr style='margin: 1rem 0'>", unsafe_allow_html=True)
        
        st.markdown("### Grouping and Filtering")
        group_mapping = {
            'Province': 'province',
            'District': 'district',
            'Subdistrict': 'subdistrict',
            'Postal Code': 'zipcode'
        }
        
        temp_selected_group = st.selectbox("Group By", list(group_mapping.keys()))
        temp_group_col = group_mapping[temp_selected_group]
        
        # Calculate metrics for region filter options
        temp_aggregated_data, _ = calculate_metrics(raw_df, temp_selected_entities, temp_data_type)
        if temp_aggregated_data is not None and temp_group_col in temp_aggregated_data:
            temp_df = temp_aggregated_data[temp_group_col]
            unique_regions = sorted(temp_df.index.unique())
            temp_selected_region = st.selectbox(
                f"Filter by {temp_selected_group}", 
                ['All'] + list(unique_regions)
            )
        else:
            temp_selected_region = 'All'
        
        # # Percentage format for min recall
        # temp_min_recall = st.slider(
        #     "Minimum %Prediction Correctness",
        #     min_value=0,
        #     max_value=100,
        #     value=0,
        #     step=5,
        #     format="%d%%"
        # ) / 100.0

        # Percentage range for prediction correctness
        temp_min_recall, temp_max_recall = st.slider(
            "%Prediction Correctness Range",
            min_value=0,
            max_value=100,
            value=(0, 100),  # default range from 0 to 100
            step=5,
            format="%d%%"
        )
        # Convert percentages to decimals
        temp_min_recall = temp_min_recall / 100.0
        temp_max_recall = temp_max_recall / 100.0
        
        # Auto-apply just once at start
        if st.session_state.get('needs_initial_apply', True):
            st.session_state.filter_settings.update({
                'data_type': temp_data_type,
                'selected_entities': temp_selected_entities,
                'selected_group': temp_selected_group,
                'selected_region': temp_selected_region,
                'min_recall': temp_min_recall,
                'max_recall': temp_max_recall,
                'group_col': temp_group_col
            })
            st.session_state.needs_initial_apply = False
            st.rerun()
        
        
        # Regular Apply Filter button for subsequent updates
        if st.button("Apply Filter"):
            st.session_state.filter_settings.update({
                'data_type': temp_data_type,
                'selected_entities': temp_selected_entities,
                'selected_group': temp_selected_group,
                'selected_region': temp_selected_region,
                'min_recall': temp_min_recall,
                'max_recall': temp_max_recall,
                'group_col': temp_group_col
            })
            # st.experimental_rerun()

    # Use the stored filter settings for visualization with safety checks
    settings = st.session_state.filter_settings
    
    # Ensure we have a valid group_col
    group_col = settings.get('group_col', 'province')
    
    # Calculate metrics based on stored settings
    aggregated_data, filtered_raw_df = calculate_metrics(
        raw_df, 
        settings['selected_entities'], 
        settings['data_type']
    )
    
    if aggregated_data is None or group_col not in aggregated_data:
        st.error("No data available for the selected filters.")
        return
    
    df = aggregated_data[group_col]
    
    # Display layout
    col1, col2 = st.columns([5, 3])
    
    with col1:
        
        
        entities_text = ", ".join([k for k, v in entity_options.items() if v in settings['selected_entities']])
        # st.markdown(f"### By {settings['selected_group']}")
        st.markdown(f"### By <span style='color: {primary_color};'> {settings['selected_group']} </span>", unsafe_allow_html=True)
        
        # Display data type and entity information
        data_type = settings['data_type']
       # Check if data_type exists in the data_types list and map it
        if data_type in data_types:
            # Find the index of the selected data_type in the data_types list
            index = data_types.index(data_type)
            
            # Map it to the corresponding description in temp_data_type_list
            mapped_data_type = temp_data_type_list[index]
        else:
            mapped_data_type = "Unknown"  # In case the data_type is not in data_types

        # Create the display text
        data_type_text = f"Address Format: {mapped_data_type}"

        # Display the result
        st.write(data_type_text)

        
 
        st.write(f"Address Components for %Correctness Calculation: {entities_text}")
        
        selected_level = None if settings['selected_region'] == 'All' else settings['selected_region']
        # m = create_optimized_map(df, group_col, selected_level, settings['min_recall'])
        m = create_optimized_map(df, group_col, selected_level, settings['min_recall'], settings['max_recall'])
        if m:
            st_folium(m
                      ,width=550
                      ,height=650
                      ,key="main_map"  # Add this to prevent reruns on zoom/pan
                      ,returned_objects=[]  # Add this to prevent returning unnecessary data
                      )

        # Function to map data_type to its corresponding description
        def assign_temp_data_type(data_type):
            if data_type == "Correct with Full Prefix":
                return "ตำบล อำเภอ"
            elif data_type == "Correct with Short Prefix":
                return "ต. อ."
            elif data_type == "Correct without Prefix":
                return "No Prefix"
            else:
                return "Others"

        # Handle 'All' case
        if temp_data_type == 'All':
            # Create a list to store random samples for each data_type
            random_samples = []

            # Loop through each unique 'data_type' in raw_df
            for data_type in raw_df['data_type'].unique():
                # Get one random sample for each data_type
                sample = raw_df[raw_df['data_type'] == data_type].sample(n=1).iloc[0]['address_with_name']
                # Append the sample with its description
                random_samples.append({
                    "Address Format": assign_temp_data_type(data_type),  # Map description
                    "Address": sample
                })

            # Convert the list of random samples into a DataFrame for display
            random_samples_df = pd.DataFrame(random_samples)

            # Reset index for a clean table display
            random_samples_df.reset_index(drop=True, inplace=True)

        else:
            # Handle specific data_type case
            if not raw_df.empty:
                # Get one random sample for the specific data_type
                random_sample = raw_df[raw_df['data_type'] == temp_data_type].sample(n=1).iloc[0]['address_with_name']
            else:
                random_sample = "No sample available for this address format."

        # Display the results
        if temp_data_type == 'All':
            st.write("Sample Address for Each Address Format")
            st.dataframe(random_samples_df, use_container_width=True, hide_index=True)
        else:
            st.write(f"Random Sample for {temp_data_type} Address Format:")
            st.write(random_sample)
    
    
    with col2:
        # filtered_df = df[df[('recall_per_row', 'mean')] >= settings['min_recall']]
        filtered_df = df[
                        (df[('recall_per_row', 'mean')] >= settings['min_recall']) & 
                        (df[('recall_per_row', 'mean')] <= settings['max_recall'])
                        ]
        if settings['selected_region'] != 'All':
            filtered_df = filtered_df.loc[[settings['selected_region']]]
 
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
    
        # Word cloud section
        # st.markdown(f"### Top 10 By {settings['selected_group']}")   
        st.markdown(f"### Top 10 By <span style='color: {primary_color};'> {settings['selected_group']} </span>", unsafe_allow_html=True)
        
        if settings['selected_region'] == 'All' and not filtered_df.empty:
            # Generate word clouds and get top/bottom data
            word_cloud_image_top, top_10 = create_word_cloud_top(filtered_df , selected_group=settings['selected_group'])
            
            if word_cloud_image_top is not None:
                # Display word clouds
                st.image(word_cloud_image_top)
                

                with st.expander(f"Most Accurate {settings['selected_group']} Details"):
                    if top_10 is not None:
                        st.markdown(f"### Most Accurate {settings['selected_group']}")
                        top_stats = pd.DataFrame({
                            'Accuracy': [f"{v:.1%}" for v in top_10['mean']],
                            'Count': top_10['count'].astype(int)
                        })
                        st.dataframe(top_stats)
                    else:
                        st.write("No data available")
                
            word_cloud_image_bottom, bottom_10 = create_word_cloud_bottom(filtered_df , selected_group=settings['selected_group'])

            if word_cloud_image_bottom is not None:
                # Display word clouds
                st.image(word_cloud_image_bottom)

                with st.expander(f"Least Accurate {settings['selected_group']} Details"):
                        if bottom_10 is not None:
                            st.markdown(f"### Least Accurate {settings['selected_group']}")
                            bottom_stats = pd.DataFrame({
                                'Accuracy': [f"{v:.1%}" for v in bottom_10['mean']],
                                'Count': bottom_10['count'].astype(int)
                            })
                            # Capitalizing the first letter of each province name
                            st.dataframe(bottom_stats)
                        else:
                            st.write("No data available")

        else:
            if settings['selected_region'] != 'All':
                st.info("Word clouds are only available when viewing all regions. Please select 'All' in the region filter to see the word clouds.")
            else:
                st.warning("No data available for word cloud visualization.")

        

if __name__ == "__main__":
    main()


## Model Part

# Load the pre-trained model
@st.cache_data
def load_model():
    model_path = 'NER_model.joblib'
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.error(f"Model file {model_path} not found.")
        return None

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
st.markdown(
    "<h1 style='font-size: 36px;'>Try out the Named Entity Recognition (NER) model yourself!</h1>",
    unsafe_allow_html=True
)

# Example input for NER analysis
example_input = "นายสมชาย เข็มกลัด 254 ถนน พญาไท แขวง วังใหม่ เขต ปทุมวัน กรุงเทพมหานคร 10330"

# Text input for user data with the example as placeholder text
user_text = st.text_area("Enter any Thai Address below:", value="", placeholder=example_input)

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


 # streamlit run 2024-11-16_test_app_lat_long_v10.py
 # /Users/kanisornunjittikul/streamlit-app/
