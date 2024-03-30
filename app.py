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

from Plotly_barchart3D import barchart

st. set_page_config(layout="wide") 
st.set_option('deprecation.showPyplotGlobalUse', False)

#plt.style.use('dark_background')
# df = pd.read_csv('data/results.csv')
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
symbol_results = symbol_results[symbol_results['method'].isin(['sax','sfa','spartan_pca_allocation'])]


results = pd.concat([results,symbol_results],ignore_index=True)
results = results.replace('sax','SAX')
results = results.replace('sfa','SFA')
results = results.replace('spartan_pca_allocation','SPARTAN')

results['method_metric'] = results['method'] + '+' + results['metric']

results = pd.pivot(results,index='dataset',columns='method_metric',values='acc')
results= results.reset_index()

bop_metrics_list = ['symbolic-l1','Euclid','BOSS','Cosine','KL-Div']

word_sizes = np.arange(2,9,1)
alphabet_sizes = np.arange(3,11,1)

def generate_dataframe(df, datasets, methods_family, metrics):
    df = df.loc[df['dataset'].isin(datasets)][[method_g + '+' + metric for metric in metrics for method_g in methods_family]]
    df = df[df.mean().sort_values().index]
    df.insert(0, 'datasets', datasets)
    return df

def plot_boxplot(df,metrics_list,datasets,method_family):
    fig = go.Figure()
    for i, cols in enumerate(df.columns[1:]):
        fig.add_trace(go.Box(y=df[cols], name=cols,
                                marker=dict(
                                    opacity=1,
                                    color='rgb(8,81,156)',
                                    outliercolor='rgba(219, 64, 82, 0.6)',
                                    line=dict(
                                        outliercolor='rgba(219, 64, 82, 0.6)',
                                        outlierwidth=2)),
                                line_color='rgb(8,81,156)'
                            ))
    fig.update_layout(showlegend=False, 
                        width=1290, 
                        height=600, 
                        template="plotly_white", 
                        font=dict(
                                size=39,
                                color="black"))
    
    fig.update_xaxes(tickfont_size=16)
    fig.update_yaxes(tickfont_size=16)
    #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
    st.plotly_chart(fig)


    st.markdown('# Bag-of-Patterns Classification Accuracy Per Dataset')
    cols_list = []
    for i, col in enumerate(df.columns):
        if i > 0:
            cols_list.append(col)
        else:
            cols_list.append(col)

    df.columns = cols_list
    AgGrid(df)

with st.sidebar:
    st.markdown('# Exploring SPARTAN')
     
    # container_metric = st.container()
    # all_metric = st.checkbox('Select all',key='all_metrics')
    # if all_metric: metrics = container_metric.multiselect('Select metric',list_measures,list_measures)
    # else: metrics = container_metric.multiselect('Select metric',list_measures)
    
    container_dataset = st.container()  
    all_cluster = st.checkbox("Select all", key='all_clusters')
    if all_cluster: cluster_size = container_dataset.multiselect('Select number of classes', sorted(list(list_num_clusters)), sorted(list(list_num_clusters)))
    else: cluster_size = container_dataset.multiselect('Select number of classes', sorted(list(list_num_clusters)))

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

    # container_method = st.container()
    # all_method = st.checkbox("Select all",key='all_method')
    # if all_method: methods_family = container_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
    # else: methods_family = container_method.multiselect('Select a group of methods',methods, key='selector_methods')

# tab_desc, tab_acc, tab_time, tab_stats, tab_analysis, tab_misconceptions, tab_ablation, tab_dataset, tab_method = st.tabs(["Description", "Evaluation", "Runtime", "Statistical Tests", "Comparative Analysis", "Misconceptions", "DNN Ablation Analysis", "Datasets", "Methods"]) 
tab_desc, tab_dataset,tab_classification_accuracy,tab_tlb = st.tabs(["Description", "Datasets","Classification Accuracy","Tightness of Lower Bound"]) 


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

with tab_classification_accuracy:
    st.markdown('# Classification Accuracy Results')
    container_method = st.container()
    all_method = st.checkbox("Select all",key='all_method')
    if all_method: methods_family = container_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
    else: methods_family = container_method.multiselect('Select a group of methods',methods, key='selector_methods')

    container_metric = st.container()
    all_metric = st.checkbox('Select all',key='all_metrics')
    if all_metric: metrics = container_metric.multiselect('Select metric',list_measures,list_measures)
    else: metrics = container_metric.multiselect('Select metric',list_measures)

    box_df = generate_dataframe(results,datasets,methods_family,metrics)
    plot_boxplot(box_df,metrics,datasets,methods_family)

with tab_tlb:
    st.markdown('# Tightness of Lower Bound Results')
    container_method = st.container()
    all_method = st.checkbox("Select all",key='all_method')
    if all_method: methods_family = container_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
    else: methods_family = container_method.multiselect('Select a group of methods',methods, key='selector_methods')

    container_tlb = st.container()
    all_metric = st.checkbox('Select all',key='all_datasets')
    if all_metric: tlb_dataset = container_tlb.selectbox('Select dataset',sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
    else: tlb_dataset = container_tlb.selectbox('Select dataset',sorted(find_datasets(cluster_size, length_size, types)))

    tlb_file = f'./data/{tlb_dataset}_tlb_results.csv'
    tlb_results = pd.read_csv(tlb_file)
    tlb_results = tlb_results.replace('sax','SAX')
    tlb_results = tlb_results.replace('sfa','SFA')
    tlb_results = tlb_results.replace('spartan','SPARTAN')

    tlb_x = tlb_results['a']
    tlb_y = tlb_results['w']
    tlb_z = tlb_results['tlb']

    fig = barchart.plotly_bar_charts_3d(tlb_x,tlb_y,tlb_z,color='x+y')
    st.plotly_chart(fig)


