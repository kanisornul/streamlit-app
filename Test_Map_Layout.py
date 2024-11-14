import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Load Thailand geospatial data (replace with your file path)
thailand_map = gpd.read_file("path_to_thailand_shapefile_or_geojson")

# Example data
data = pd.DataFrame({
    'province': ['Bangkok', 'Chiang Mai', 'Phuket'], 
    'density': [200, 50, 120]
})

# Merge data with geospatial data
thailand_map = thailand_map.merge(data, left_on="province_name_column", right_on="province", how="left")

# Streamlit app layout
st.title("Thailand Density Map")

# 1. Filter Section
with st.sidebar:
    st.header("Filter")
    # Add filters as needed, for example:
    selected_provinces = st.multiselect("Select Provinces", options=data['province'].unique())

# 2. Thailand Map Section
with st.container():
    st.header("Thailand Map")
    
    # Plot map
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    thailand_map.plot(column='density', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    ax.set_title("Density by Province", fontdict={'fontsize': '15', 'fontweight' : '3'})
    ax.set_axis_off()

    # Display plot in Streamlit
    st.pyplot(fig)

# 3. Additional Information Section
with st.container():
    st.header("Additional Information")
    
    # Section 1: Description or summary
    st.subheader("Summary")
    st.write("This section provides a summary of the data.")

    # Section 2: Statistics
    st.subheader("Statistics")
    st.write("This section contains statistical information.")

    # Section 3: Other Details
    st.subheader("Other Details")
    st.write("Additional details or insights about the data.")
