# Interactive Anomaly detection Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()

Identify anomalous Windows network logon sessions using an Isolation Forest algorithm with interactive anomaly detection dashboard

## Problem Statement

- **Dynamic User Logon Behavior Analysis:** Traditional methods struggle to accurately detect unusual logon activities due to the dynamic nature of user logon behaviors, which include multiple attributes like method and authentication type.
- **Complex Logon Patterns:** User logon patterns are recorded with various properties (e.g., interactive/remote logon, NTLM/Kerberos authentication, source IP, location, destination host), changing with the environment.
- **Isolation Forest for Anomaly Detection:** Utilizes the Isolation Forest machine learning algorithm to identify anomalies in logon activities by isolating data points based on random thresholds and recursively splitting the dataset.
- **Effective Anomaly Identification:** Determines anomalies by calculating the average path length required to isolate a sample within the Isolation Forest, where shorter path lengths suggest anomalous behavior.
- **Noise Reduction in Detections:** Reduces traditional detection noise by focusing on complex, evolving user logon patterns rather than static logon properties, offering a sophisticated approach to identifying security threats.

## Architecture
![raw_image](https://raw.github.com/ashwin-patil/streamlit-anomaly-demo/master/images/DataFlowDiagram.png)


## Docker instruction
if you wish to host this locally/in-house, you can use below instructions to build docker images and host it. For more detailed instructions, check out Streamlit docs. [Deploy Streamlit using Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

Build image

`docker build -t streamlit-anomaly-demo .`

Run the docker container

`docker run -p 8501:8501 streamlit-anomaly-demo`

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.