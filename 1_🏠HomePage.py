import streamlit as st
import pandas as pd

from pathlib import Path


def main() -> None:
    st.title(":mag_right: Interactive Anomaly detection using Isolation Forest")

    with st.expander("Expand to Read more about the Anomaly detection"):
        # Load and display the README.md file
        try:
            with open('README.md', 'r', encoding='utf-8') as file:
                readme_contents = file.read()
                st.markdown(readme_contents, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error('README.md file not found.')

    st.success(":point_left: Select a page on left side bar to naviagate pages")


if __name__ == "__main__":
    st.set_page_config(
        "Interactive Anomaly detection",
        "🔎",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()