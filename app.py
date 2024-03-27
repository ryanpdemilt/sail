from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from constant import *
from constant import compute_CD
from st_aggrid import AgGrid
import plotly.graph_objects as go
from statistical_test import graph_ranks

st. set_page_config(layout="wide") 
st.set_option('deprecation.showPyplotGlobalUse', False)

#plt.style.use('dark_background')
df = pd.read_csv('data/results.csv')
characteristics_df = pd.read_csv('data/characteristics.csv')
characteristics_df.reset_index()
characteristics_df.drop(columns=characteristics_df.columns[0], axis=1, inplace=True)
characteristics_df.columns = ['Name', 'NumOfTrainingSamples', 'NumOfTestingSample', 'NumOfSamples', 'SeqLength', 'NumOfClasses', 'Type']
characteristics_df = characteristics_df[['Name', 'NumOfSamples', 'SeqLength', 'NumOfClasses', 'Type']]

# tab_desc, tab_acc, tab_time, tab_stats, tab_analysis, tab_misconceptions, tab_ablation, tab_dataset, tab_method = st.tabs(["Description", "Evaluation", "Runtime", "Statistical Tests", "Comparative Analysis", "Misconceptions", "DNN Ablation Analysis", "Datasets", "Methods"]) 
tab_desc, tab_dataset = st.tabs(["Description", "Datasets"]) 

with tab_desc:
    st.markdown('# SPARTAN')
    st.markdown(description_intro1)
    background = Image.open('./data/spartan_pipeline.png')
    col1, col2, col3 = st.columns([1.2, 5, 0.2])
    col2.image(background, width=900, caption='Overview of the SPARTAN representation method.')
    # st.markdown(description_intro2)
    # background = Image.open('./data/taxonomy.png')
    # col1, col2, col3 = st.columns([1.2, 5, 0.2])
    # col2.image(background, width=900, caption='Taxonomy of time-series clustering methods in Odyssey.')

with tab_dataset:
    st.markdown('# Dataset Description')
    st.markdown(text_description_dataset)
    AgGrid(characteristics_df)