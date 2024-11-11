import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt

# Load your dataset
merged_df = pd.read_csv(f'/Users/kanisornunjittikul/streamlit-app/streamlit-app/final_data_present3.csv')

# Create filtering widgets
def create_filtering_widgets():
    # Define the columns and their display names for entity checkboxes
    flag_columns = ['m_subdistrict_flag', 'm_district_flag', 'm_province_flag', 'm_zipcode_flag']
    flag_column_labels = {col: col.split('_')[1].capitalize() for col in flag_columns}
    
    # Define data type options
    data_type_options = merged_df['data_type'].dropna().unique()  # dropna() to handle null values
    
    # Define grouping options
    grouping_options = [
        'No Grouping',
        'Province',
        'District',
        'Subdistrict',
        'Postal Code'
    ]
    
    # Streamlit widgets
    entity_checkboxes = {label: st.checkbox(label, value=True) for label in flag_column_labels.values()}
    
    select_all_data_type = st.checkbox("Select All Data Type", value=True)
    data_type_checkboxes = {option: st.checkbox(option, value=True) for option in data_type_options}
    
    grouping_dropdown = st.selectbox('Group by:', grouping_options, index=0)
    map_dropdown = st.selectbox('Color by:', list(flag_column_labels.values()), index=0)

    return {
        'entity_checkboxes': entity_checkboxes,
        'select_all_data_type': select_all_data_type,
        'data_type_checkboxes': data_type_checkboxes,
        'grouping_dropdown': grouping_dropdown,
        'map_dropdown': map_dropdown,
        'flag_columns': flag_columns,
        'flag_column_labels': flag_column_labels
    }

# Create map function
def make_map(selected_column, filtered_df=None):
    """Create map with the selected column and filtered data"""
    # Example of creating a map with folium (assumes you have geographic data)
    m = folium.Map(location=[13.7563, 100.5018], zoom_start=6)
    folium.GeoJson(filtered_df).add_to(m)  # Assuming filtered_df contains geo data
    return m

# Grouping and metrics calculation
def calculate_group_metrics(df, group_col):
    """Calculate metrics for grouped data"""
    group_metrics = df.groupby(group_col).agg({
        'recall_per_row': ['count', 'mean', 'median', 'std'],
    }).round(4)
    
    group_metrics.columns = ['Count', 'Mean Recall', 'Median Recall', 'Std Dev']
    return group_metrics

# Display output (metrics and map)
def update_display(widgets_dict, metrics_output, map_output):
    """Update both metrics and map based on current selections"""
    
    # Get selected entities
    selected_columns = [col for col, checkbox in widgets_dict['entity_checkboxes'].items() if checkbox]
    
    # Get selected data types
    selected_data_types = [option for option, checkbox in widgets_dict['data_type_checkboxes'].items() if checkbox]
    
    # Apply filters
    filtered_df = merged_df.copy()
    if selected_data_types:
        filtered_df = filtered_df[filtered_df['data_type'].isin(selected_data_types)]
    
    # Calculate recall on filtered data
    filtered_df['recall_per_row'] = (filtered_df[selected_columns].sum(axis=1) / 
                                      filtered_df[selected_columns].notnull().sum(axis=1))
    
    # Handle grouping and display metrics
    grouping_level = widgets_dict['grouping_dropdown']
    if grouping_level != 'No Grouping':
        group_mapping = {
            'Province': 'province',
            'District': 'district',
            'Subdistrict': 'subdistrict',
            'Postal Code': 'zipcode'
        }
        group_col = group_mapping[grouping_level]
        group_metrics = calculate_group_metrics(filtered_df, group_col)
        
        st.write(f"Recall Statistics Grouped by {grouping_level}:")
        st.dataframe(group_metrics)
    else:
        st.write("Overall Recall Statistics:")
        st.write(f"Number of records: {len(filtered_df)}")
        st.write(f"Mean Recall: {filtered_df['recall_per_row'].mean():.4f}")
        st.write(f"Median Recall: {filtered_df['recall_per_row'].median():.4f}")
        st.write(f"Std Recall: {filtered_df['recall_per_row'].std():.4f}")
        
    # Update map
    map_output.map(make_map(widgets_dict['map_dropdown'], filtered_df))

def main():
    # Create all widgets
    widgets_dict = create_filtering_widgets()
    
    # Create output containers
    metrics_output = st.empty()
    map_output = st.empty()

    # Initial update
    update_display(widgets_dict, metrics_output, map_output)

    # Update on changes (Streamlit automatically updates on widget interaction)
    st.button("Update", on_click=update_display, args=(widgets_dict, metrics_output, map_output))

# Run the application
if __name__ == "__main__":
    st.set_page_config(layout="wide")  # Set wide layout for Streamlit
    main()
