import streamlit as st
import sys
import subprocess
import os

# Function to install missing packages
def install_missing_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "geopandas==0.14.1",
                             "matplotlib==3.8.2",
                             "pandas==2.2.0",
                             "numpy==1.26.3",
                             "pyproj==3.6.1",
                             "shapely==2.0.2",
                             "fiona==1.9.5",
                             "rtree==1.1.0"])
        return True
    except Exception as e:
        st.error(f"Failed to install packages: {str(e)}")
        return False

# Try importing required packages
try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import pandas as pd
    from pathlib import Path
    import pickle
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
            max-width: 220px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Safe data loading with error handling
@st.cache_data
def load_geojson():
    try:
        # First try the data directory
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

@st.cache_data
def create_map(_data, selected_provinces=None):
    try:
        # Create figure with subplot and toolbar
        fig = plt.figure(figsize=(8, 10), dpi=100)
        ax = fig.add_subplot(111)
        
        # Enable the navigation toolbar
        plt.rcParams['toolbar'] = 'toolmanager'
        
        if selected_provinces:
            _data.plot(ax=ax, color='#E6E6E6', edgecolor='white', linewidth=0.5)
            _data[_data['NAME_1'].isin(selected_provinces)].plot(
                ax=ax,
                color='#ADD8E6',
                edgecolor='white',
                linewidth=0.5
            )
        else:
            _data.plot(ax=ax, color='#E6E6E6', edgecolor='white', linewidth=0.5)
        
        bounds = _data.total_bounds
        ax.set_xlim(bounds[0] - 0.5, bounds[2] + 0.5)
        ax.set_ylim(bounds[1] - 0.5, bounds[3] + 0.5)
        
        ax.set_title("Thailand Provinces", pad=10, fontsize=12)
        
        # Add zoom and pan instructions
        ax.text(0.02, 0.02, 
                "Use mousewheel to zoom\nClick and drag to pan",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.tight_layout(pad=0.5)
        
        # Enable interactive features
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        
        return fig
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
            
            # Add reset view button
            if st.button("Reset View"):
                st.session_state['map_view'] = None

        create_metrics_section(thailand_map, selected_provinces)
        
        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            fig = create_map(thailand_map, selected_provinces)
            if fig is not None:
                # Create the plot with interactive features enabled
                st.pyplot(fig, use_container_width=True)

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
