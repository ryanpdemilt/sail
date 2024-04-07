# SAIL: A Voyage to Symbolic Representation Solutions for Time-Series Analysis

Symbolic Representation, a dimensionality reduction technique that transforms time series into a string of discrete symbols, has received increasing attention in downstream tasks such as similarity search, indexing, anomaly detection, and classification.
Despite vast progress in this area, the majority of solutions still rely on a well-established method proposed two decades ago for its simplicity, despite other potential solutions with stronger representation power.
Moreover, from the existing literature, there is a noticeable absence of a comprehensive study in this domain, highlighting a need for more in-depth investigation.
Motivated by the aforementioned issues, we propose SAIL, a modular web engine serving two purposes: (i) to provide the first comprehensive evaluation studies on 7 state-of-the-art symbolic representation methods over 128 time-series datasets, the largest study in this area;
(ii) to demonstrate the superiority of a recently proposed solution, SPARTAN, that mitigates the non-negligible drawbacks of the uniform balance in symbolic subspaces, which assumes a single alphabet size for each symbol.
Through the interactive interface, SAIL facilitates users to explore, visualize, and comprehend the quantitative assessment results on various methods, datasets, and metrics.
From diverse scenarios, SPARTAN has demonstrated superior performance across different dimensions.
Overall, SAIL not only facilitates the most comprehensive evaluation study in this field, but also delivers new insights and concrete solutions, laying the groundwork for the next-generation symbolic representation solutions.

## Exploring Symbolic Representations

The demo is hosted online using streamlit at this [link](https://sail-symbolic-exploration.streamlit.app/)

If desired SAIL can also be hosted locally for offline viewing and experimentation, simply perform the following steps to clone the repository and install the required packages to host the app on a local machine.

```sh
git clone https://github.com/ryanpdemilt/sail.git

cd sail

pip install -r requirements.txt

python3 -m streamlit run app.py
```