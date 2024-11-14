import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

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
        /* Optimize map container */
        .map-container {
            max-height: 600px;
            margin: auto;
        }
        /* Make expanders more compact */
        .streamlit-expanderHeader {
            padding: 0.3rem;
        }
        /* Optimize title */
        h1 {
            margin-top: 0;
            padding-top: 0.3rem;
            margin-bottom: 0.5rem;
            font-size: 1.8rem !important;
        }
        /* Custom sidebar width */
        [data-testid="stSidebar"] {
            min-width: 220px !important;
            max-width: 220px !important;
        }
        /* Metric containers */
        [data-testid="metric-container"] {
            padding: 0.5rem !important;
        }
        /* Adjust column gaps */
        [data-testid="column"] {
            padding: 0.5rem !important;
        }
        /* Make dataframe more compact */
        .stDataFrame {
            max-height: 250px;
            overflow-y: auto;
        }
        /* Reduce padding in expanders */
        .streamlit-expanderContent {
            padding: 0.5rem !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_geojson():
    try:
        cache_file = Path("thailand_map_cache.pkl")
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            data = gpd.read_file("thailand-provinces.geojson")
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def create_map(_data, selected_provinces=None):
    # Adjusted figure size for better desktop fit
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=100)  # Modified aspect ratio for Thailand
    
    # Plot with adjusted boundaries
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
    
    # Adjust boundaries to focus on Thailand
    bounds = _data.total_bounds
    ax.set_xlim(bounds[0] - 0.5, bounds[2] + 0.5)
    ax.set_ylim(bounds[1] - 0.5, bounds[3] + 0.5)
    
    ax.set_title("Thailand Provinces", pad=10, fontsize=12)
    ax.set_axis_off()
    plt.tight_layout(pad=0.5)
    
    return fig

def create_metrics_section(thailand_map, selected_provinces):
    cols = st.columns([1, 1, 1, 2])  # Adjusted column ratio
    
    with cols[0]:
        st.metric("Total Provinces", f"{len(thailand_map)}")
    with cols[1]:
        st.metric("Selected", f"{len(selected_provinces) if selected_provinces else 0}")
    with cols[2]:
        percentage = (len(selected_provinces) / len(thailand_map) * 100) if selected_provinces else 0
        st.metric("Selection %", f"{percentage:.1f}%")

def main():
    thailand_map = load_geojson()
    if thailand_map is None:
        return

    # Sidebar
    with st.sidebar:
        st.header("Filters")
        search_term = st.text_input("Search Provinces", "")
        
        # Filter provinces based on search
        province_options = sorted(thailand_map['NAME_1'].unique())
        if search_term:
            province_options = [p for p in province_options 
                              if search_term.lower() in p.lower()]
        
        selected_provinces = st.multiselect(
            "Select Provinces",
            options=province_options,
            key='province_selector'
        )

    # Main content
    st.title("Thailand Map Explorer")
    create_metrics_section(thailand_map, selected_provinces)
    
    # Main content columns with adjusted ratios
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        # Create map with adjusted size
        fig = create_map(thailand_map, selected_provinces)
        st.pyplot(fig, use_container_width=True)

    with col2:
        # Province Details
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

        # Quick Statistics
        with st.expander("Quick Statistics", expanded=True):
            if selected_provinces:
                st.write("Selected Provinces:")
                for idx, province in enumerate(sorted(selected_provinces), 1):
                    st.write(f"{idx}. {province}")
            else:
                st.info("Select provinces to view statistics")

        # Help Section
        with st.expander("Help", expanded=False):
            st.write("### Quick Guide")
            st.write("1. Use the search box to filter provinces")
            st.write("2. Select multiple provinces from the dropdown")
            st.write("3. View details and statistics in the panels")

if __name__ == "__main__":
    main()
