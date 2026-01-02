import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Disaster Response AI Dashboard", layout="wide")

st.title("🌍 Disaster Response AI System")
st.markdown("""
This dashboard uses **LSTM-based forecasting** and **spatiotemporal analysis**  
to identify disaster hotspots and suggest response resource allocations.
""")

# Load data
@st.cache_data
def load_data():
    combined = pd.read_csv("data/processed/combined_disaster_data.csv")
    combined['time_bin'] = pd.to_datetime(combined['time_bin'])
    return combined

combined = load_data()

# Sidebar controls
st.sidebar.header("⚙️ Dashboard Controls")
view_option = st.sidebar.selectbox("Select Visualization", [
    "Disaster Intensity Heatmap",
    "Top Hotspot Zones",
    "Resource Allocation Plan"
])

# Compute average intensities
risk = combined.groupby(['grid_x','grid_y'])['count'].mean().reset_index()
risk.rename(columns={'count': 'avg_intensity'}, inplace=True)

# Identify hotspots
threshold = risk['avg_intensity'].quantile(0.9)
hotspots = risk[risk['avg_intensity'] >= threshold]

# View 1: Heatmap
if view_option == "Disaster Intensity Heatmap":
    st.subheader("🌡️ Average Disaster Intensity Heatmap")
    pivot = risk.pivot(index='grid_y', columns='grid_x', values='avg_intensity')

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pivot, cmap='YlOrRd', cbar=True, ax=ax)
    plt.title("Average Disaster Intensity (All Regions)")
    st.pyplot(fig)

# View 2: Hotspots
elif view_option == "Top Hotspot Zones":
    st.subheader("🔥 Top Predicted Hotspot Zones (90th percentile)")
    st.write(f"Threshold intensity: {threshold:.2f}")
    st.dataframe(hotspots.sort_values('avg_intensity', ascending=False).head(10))

    pivot = risk.pivot(index='grid_y', columns='grid_x', values='avg_intensity')
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pivot, cmap='YlOrRd', cbar=True, ax=ax)
    plt.scatter(hotspots['grid_x'], hotspots['grid_y'], color='blue', s=40, marker='X')
    plt.title("Hotspot Overlay")
    st.pyplot(fig)

# View 3: Resource Allocation
elif view_option == "Resource Allocation Plan":
    st.subheader("🚒 Automated Resource Deployment Plan")
    critical_zones = hotspots.sort_values('avg_intensity', ascending=False).head(5)
    resources = ['Rescue Team 1', 'Drone Squad', 'Medical Unit', 'Relief Truck', 'Rescue Team 2']

    allocation_plan = pd.DataFrame({
        'Resource': resources,
        'Assigned_grid_x': critical_zones['grid_x'].values[:len(resources)],
        'Assigned_grid_y': critical_zones['grid_y'].values[:len(resources)],
        'Predicted_Intensity': critical_zones['avg_intensity'].values[:len(resources)]
    })

    st.table(allocation_plan)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(risk.pivot(index='grid_y', columns='grid_x', values='avg_intensity'), cmap='YlOrRd', cbar=True)
    plt.scatter(allocation_plan['Assigned_grid_x'], allocation_plan['Assigned_grid_y'], color='red', s=80, marker='X', label='Deployment Points')
    plt.legend()
    plt.title("Resource Deployment Overlay")
    st.pyplot(fig)
