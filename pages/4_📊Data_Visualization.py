import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest



def main():
    
    demo_data = pd.read_csv('data/iforest-demo-data.csv')
    data = outliers

    st.title('Interactive Widgets for Data Visualization')

    # Side bar widgets
    with st.sidebar:
        st.header('Isolation Forest Parameters')
        
        # # n_estimators slider
        # n_estimators = st.slider('n_estimators', min_value=1, max_value=100, value=50)
        
        # # contamination dropdown
        # contamination = st.select_slider('contamination', options=[float(i)/100 for i in range(0, 31, 1)])
        
        # # max_features slider
        # max_features = st.slider('max_features', min_value=1, max_value=6, value=3)
        
        # features multi-select
        features = st.multiselect('Select Features', options=[
            "FailedLogons_zscore",
            "SuccessfulLogons_zscore",
            "ComputersSuccessfulAccess_zscore",
            "ComputerDomainsSuccessfulAccess_zscore",  
            "SrcIpSuccessfulAccess_zscore",
            "SrcHostNameSuccessfulAccess_zscore" 
        ], default=[
        "FailedLogons_zscore",
        "SuccessfulLogons_zscore",
        "ComputersSuccessfulAccess_zscore",
        "ComputerDomainsSuccessfulAccess_zscore",  
        "SrcIpSuccessfulAccess_zscore",
        "SrcHostNameSuccessfulAccess_zscore" 
    ])

    # specify the metrics column names to be modelled
    features = [
        "FailedLogons_zscore",
        "SuccessfulLogons_zscore",
        "ComputersSuccessfulAccess_zscore",
        "ComputerDomainsSuccessfulAccess",
        "SrcIpSuccessfulAccess_zscore",
        "SrcHostNameSuccessfulAccess",
    ]
    pca3 = PCA(n_components=3)  # Reduce to k=3 dimensions
    scaler = StandardScaler()
    # normalize the metrics
    X = scaler.fit_transform(data[features])
    X_reduce = pca3.fit_transform(X)


    # Add destination user to label
    data["labels"] = np.where(
        data["anomaly"] == -1,
        data["Date"].astype("str")
        + " "
        + data["DstDomain"].astype("str")
        + " DstUser "
        + data["DstUser"].astype("str"),
        "non-anomalous",
    )

    total_var = pca3.explained_variance_ratio_.sum() * 100
    fig = px.scatter_3d(
        X_reduce,
        x=0,
        y=1,
        z=2,
        color=data["labels"],
        title=f"Total Explained Variance: {total_var:.2f}%",
        labels={
            "0": "Principal Component 1",
            "1": "Principal Component 2",
            "2": "Principal Component 3",
        },
    )
    fig.show()

if __name__ == "__main__":
    st.set_page_config(
        "Interactive data visualization",
        "🧩",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()