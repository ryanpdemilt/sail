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

results = pd.read_csv('./data/bop_results.csv')
results = results.replace('hist_euclidean','Euclid')
results = results.replace('boss','BOSS')
results = results.replace('cosine_similarity','Cosine')
results = results.replace('kl','KL-Div')

symbol_results = pd.read_csv('./data/symbol_results.csv')
symbol_results['metric'] = 'symbolic-l1'

results = pd.concat([results,symbol_results],ignore_index=True)

bop_metrics_list = ['symbol','Euclid','BOSS','Cosine','KL-Div']

def plot_boxplot(df,metrics_list,methods,datasets):
    pass

with st.sidebar:
    st.markdown('# SPARTAN Exploration') 
    metric_name = st.selectbox('Pick an assessment measure', list_measures) 
    
    container_dataset = st.container()  
    all_cluster = st.checkbox("Select all", key='all_clusters')
    if all_cluster: cluster_size = container_dataset.multiselect('Select  size', sorted(list(list_num_clusters)), sorted(list(list_num_clusters)))
    else: cluster_size = container_dataset.multiselect('Select cluster size', sorted(list(list_num_clusters)))

    container_dataset = st.container()  
    all_length = st.checkbox("Select all", key='all_lengths')
    if all_length: length_size = container_dataset.multiselect('Select sequence length size', sorted(list(list_seq_length)), sorted(list(list_seq_length)))
    else: length_size = container_dataset.multiselect('Select length sequence size', sorted(list(list_seq_length)))

    container_dataset = st.container()  
    all_type = st.checkbox("Select all", key='all_types')
    if all_type: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)), sorted(list(list_type)))
    else: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)))
   
    container_dataset = st.container()  
    all_dataset = st.checkbox("Select all", key='all_dataset')
    if all_dataset: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
    else: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types)))

    container_method = st.container()
    all_method = st.checkbox("Select all",key='all_method')
    if all_method: methods_family = container_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
    else: methods_family = container_method.multiselect('Select a group of methods',methods, key='selector_methods')

# tab_desc, tab_acc, tab_time, tab_stats, tab_analysis, tab_misconceptions, tab_ablation, tab_dataset, tab_method = st.tabs(["Description", "Evaluation", "Runtime", "Statistical Tests", "Comparative Analysis", "Misconceptions", "DNN Ablation Analysis", "Datasets", "Methods"]) 
tab_desc, tab_dataset,tab_bop_accuracy = st.tabs(["Description", "Datasets","BOP Accuracy"]) 


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

with tab_bop_accuracy:
    st.markdown('# Bag-Of-Patterns Accuracy Results')
