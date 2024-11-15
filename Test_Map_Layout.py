import streamlit as st
import sys
import subprocess
import os

# Function to install missing packages
def install_missing_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "geopandas==0.14.1",
                             "folium==0.15.1",
                             "pandas==2.2.0",
                             "numpy==1.26.3",
                             "pyproj==3.6.1",
                             "shapely==2.0.2",
                             "fiona==1.9.5",
                             "rtree==1.1.0",
                             "streamlit-folium==0.18.0"])
        return True
    except Exception as e:
        st.error(f"Failed to install packages: {str(e)}")
        return False

# Try importing required packages
try:
    import geopandas as gpd
    import folium
    from streamlit_folium import folium_static
    import pandas as pd
    from pathlib import Path
except ImportError as e:
    st.error(f"Missing required packages: {str(e)}")
    if install_missing_packages():
        st.experimental_rerun()
    else:
        st.stop()

# Configuration for desktop display
st.set_page_config(
    page_title="Thailand Map Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for better desktop fit
st.markdown("""
    <style>
        .main > div {
            padding-top: 0.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stSidebar > div {
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-testid="stSidebar"] {
            min-width: 220px !important;
            max-width: 300px !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_geojson():
    try:
        data_paths = [
            "thailand-provinces.geojson",
            "data/thailand-provinces.geojson",
            "/mount/src/streamlit-app/thailand-provinces.geojson",
            os.path.join(os.path.dirname(__file__), "thailand-provinces.geojson")
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                return gpd.read_file(path)
        
        st.error("Could not find the GeoJSON file. Please check the file location.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_folium_map(data, selected_provinces=None):
    try:
        # Get the center of Thailand for the initial view
        center_lat = data.geometry.centroid.y.mean()
        center_lon = data.geometry.centroid.x.mean()
        
        # Create the base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='CartoDB positron'
        )
        
        # Function to style the provinces
        def style_function(feature):
            province_name = feature['properties']['NAME_1']
            if selected_provinces and province_name in selected_provinces:
                return {
                    'fillColor': '#ADD8E6',
                    'fillOpacity': 0.7,
                    'color': 'white',
                    'weight': 1,
                }
            return {
                'fillColor': '#E6E6E6',
                'fillOpacity': 0.7,
                'color': 'white',
                'weight': 1,
            }
        
        # Add hover functionality
        def highlight_function(feature):
            return {
                'fillOpacity': 0.9,
                'weight': 2,
            }
        
        # Add GeoJson layer with tooltips
        folium.GeoJson(
            data.__geo_interface__,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['NAME_1'],
                aliases=['Province:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
            )
        ).add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def create_metrics_section(thailand_map, selected_provinces):
    try:
        cols = st.columns([1, 1, 1, 2])
        
        with cols[0]:
            st.metric("Total Provinces", f"{len(thailand_map)}")
        with cols[1]:
            st.metric("Selected", f"{len(selected_provinces) if selected_provinces else 0}")
        with cols[2]:
            percentage = (len(selected_provinces) / len(thailand_map) * 100) if selected_provinces else 0
            st.metric("Selection %", f"{percentage:.1f}%")
    except Exception as e:
        st.error(f"Error creating metrics: {str(e)}")

def main():
    try:
        st.title("Thailand Map Explorer")
        
        thailand_map = load_geojson()
        if thailand_map is None:
            st.warning("Please ensure the GeoJSON file is in the correct location.")
            st.stop()

        # Sidebar
        with st.sidebar:
            st.header("Filters")
            search_term = st.text_input("Search Provinces", "")
            
            province_options = sorted(thailand_map['NAME_1'].unique())
            if search_term:
                province_options = [p for p in province_options 
                                  if search_term.lower() in p.lower()]
            
            selected_provinces = st.multiselect(
                "Select Provinces",
                options=province_options,
                key='province_selector'
            )

        create_metrics_section(thailand_map, selected_provinces)
        
        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            folium_map = create_folium_map(thailand_map, selected_provinces)
            if folium_map is not None:
                # Display the map with folium_static
                folium_static(folium_map, width=700, height=500)
                
                # Add instructions
                st.caption("Map Controls:")
                st.caption("• Use +/- buttons or mousewheel to zoom")
                st.caption("• Click and drag to pan")
                st.caption("• Hover over provinces to see names")

        with col2:
            with st.expander("Province Details", expanded=True):
                if selected_provinces:
                    filtered_data = thailand_map[thailand_map['NAME_1'].isin(selected_provinces)]
                    st.dataframe(
                        filtered_data[['NAME_1']].sort_values('NAME_1'),
                        use_container_width=True,
                        height=150
                    )
                else:
                    st.info("Select provinces to view details")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
