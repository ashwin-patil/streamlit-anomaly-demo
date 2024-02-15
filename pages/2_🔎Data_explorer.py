import streamlit as st
from st_aggrid import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import pandas as pd

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


def main():
    st.title('Interactive Data Table using Streamlit AggGrid')

    # File uploader allows user to add their own CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    demo_data = pd.read_csv('data/iforest-demo-data.csv')

    if uploaded_file is not None:
        # Read the CSV file into a pandas dataframe
        df = pd.read_csv(uploaded_file)
        
        # Display the dataframe using streamlit-aggrid
        st.write("Interactive Data Table:")
        aggrid_interactive_table(df=demo_data)

if __name__ == "__main__":
    st.set_page_config(
        "Interactive data explorer",
        "🔎",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()