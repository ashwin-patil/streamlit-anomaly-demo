import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode


def aggrid_interactive_table(df: pd.DataFrame):
    """Source : https://github.com/streamlit/example-app-interactive-table
    Creates an st-aggrid interactive table based on a dataframe.
    Args:
        df (pd.DataFrame]): Source dataframe
    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="balham",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

def get_zscore(value, mean, std):
    # calculate z-score or number of standard deviations from mean
    if (
        std == 0
        or std is None
        or str(std).lower() in ["nan", "none", "null"]
        or mean is None
    ):
        if value == 0.0:
            return 0.0
        elif value != 0:
            return np.log10(value + 1)
    ans = (value - mean) / std
    # only interested in increases
    ans = max(0.0, ans)
    # take log to dampen numbers
    ans = np.log10(ans + 1)
    return float(ans)

def apply_isolation_forest(df, n_estimators, contamination=0.01):
    """Applies Isolation Forest to a given dataset and returns the predicted anomalies."""
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        max_features=6,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    clf.fit(df.values)
    pred = clf.predict(df.values)
    scores = clf.decision_function(df.values)
    return clf, pred, scores

def main():
    
    demo_data = pd.read_csv('data/iforest-demo-data.csv')
    data = demo_data.copy()

    st.title('Interactive Widgets for Isolation Forest Parameters')

    # Side bar widgets
    with st.sidebar:
        st.header('Isolation Forest Parameters')
        
        # n_estimators slider
        n_estimators = st.slider('n_estimators', min_value=1, max_value=100, value=50)
        
        # contamination dropdown
        contamination = st.select_slider('contamination', options=[float(i)/100 for i in range(0, 31, 1)])
        
        # max_features slider
        max_features = st.slider('max_features', min_value=1, max_value=6, value=3)
        
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
    
    # Display selected parameters
    st.write("### Selected Parameters")
    st.write(f"- **n_estimators**: {n_estimators}")
    st.write(f"- **contamination**: {contamination}")
    st.write(f"- **max_features**: {max_features}")
    st.write(f"- **features**: {features}")

    zscore_columns = [
        "FailedLogons",
        "SuccessfulLogons",
        "ComputersSuccessfulAccess",
        "SrcIpSuccessfulAccess",
    ]
    means = [x + "_mean" for x in zscore_columns]
    stds = [x + "_std" for x in zscore_columns]
    zscores = [x + "_zscore" for x in zscore_columns]

    ind = ["DstDomain", "DstUser", "Date"]

    zscore = data[zscore_columns + ind]
    zscore = zscore.fillna(0)

    # getting means for user, domain and logon type combination
    zscore[means] = zscore.groupby(["DstDomain", "DstUser"])[zscore_columns].transform(
        "mean"
    )

    # getting standard deviation for user, domain and logon type combination
    zscore[stds] = zscore.groupby(["DstDomain", "DstUser"])[zscore_columns].transform(
        "std", ddof=1
    )

    zscore = zscore.drop_duplicates(["DstDomain", "DstUser"])

    zscore = zscore[means + stds + ["DstDomain", "DstUser"]]

    data = data.merge(zscore, how="left", on=["DstDomain", "DstUser"])

    # Calculating z scores
    for column in zscore_columns:
        data[f"{column}_zscore"] = data.apply(
            lambda row: get_zscore(
                row[f"{column}"], row[f"{column}_mean"], row[f"{column}_std"]
            ),
            axis=1,
        )

    # specify the metrics column names to be modelled
    features = [
        "FailedLogons_zscore",
        "SuccessfulLogons_zscore",
        "ComputersSuccessfulAccess_zscore",
        "ComputerDomainsSuccessfulAccess",
        "SrcIpSuccessfulAccess_zscore",
        "SrcHostNameSuccessfulAccess",
    ]

    data[features] = data[features].fillna(0)

    X = data[features].copy()
    if X.shape[0] < 500:
        n_estimators = len(features) * 4 + X.shape[0] * 2
    else:
        n_estimators = 100

    # n_estimators = 100
    print("Number of trees", n_estimators)

    clf, pred, scores = apply_isolation_forest(X, n_estimators, contamination=0.01)
    data["anomaly"] = pred
    data["score"] = scores * -1
    # excluding users who do not have any successful logon history.
    data = data.loc[data["SuccessfulLogons"] > 0]
    outliers = data.loc[data["anomaly"] == -1]
    outlier_index = list(outliers.index)
    # print(f"Outliers at indexes: {outlier_index}")
    # # Find the number of anomalies and normal points here points classified -1 are anomalous
    # print(data["anomaly"].value_counts())
    # Display the dataframe using streamlit-aggrid
    st.write("Top anomalies by score (top most has highest anomaly score and so on")
    
    outliers = outliers.sort_values(by=["score"], ascending=False)
    aggrid_interactive_table(df=outliers)


if __name__ == "__main__":
    st.set_page_config(
        "Interactive data modelling",
        "🧩",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()