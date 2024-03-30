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

import os

from barchart import plotly_bar_charts_3d

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

bop_metrics_list = ['Euclid','BOSS','Cosine','KL-Div']
onenn_metrics_list = ['symbolic-l1']
onenn_results = pd.read_csv('./data/a4_w12_all_methods.csv')

onenn_results = onenn_results.rename(columns={'dataset_name':'dataset','classifier_name':'method','accuracy':'acc'})

onenn_results['method_metric'] = onenn_results['method'] + '+' + 'symbolic_l1'

onenn_results = pd.pivot(onenn_results,index='dataset',columns = 'method_metric',values='acc')
onenn_results= onenn_results.reset_index()

onenn_methods_list =['SAX','SFA','SPARTAN','SAX-DR','SAX_VFD','TFSAX','1d-SAX','ESAX']



word_sizes = np.arange(2,9,1)
alphabet_sizes = np.arange(3,11,1)

tlb_file = './data/tlb/'

tlb_files = os.listdir(tlb_file)
tlb_dfs = {}
for file in tlb_files:
    dset = file.split('_')[0]
    csv_file = tlb_file + file
    tlb_dfs[dset] = pd.read_csv(csv_file)


def generate_dataframe(df, datasets, methods_family, metrics):
    df = df.loc[df['dataset'].isin(datasets)][[method_g + '+' + metric for metric in metrics for method_g in methods_family]]
    df = df[df.mean().sort_values().index]
    df.insert(0, 'datasets', datasets)
    return df

def plot_boxplot(df,metrics_list,datasets,method_family,key='table_bop'):
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
    AgGrid(df,key=key)

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
tab_desc, tab_dataset,tab_1nn_classification,tab_classification_accuracy,tab_tlb = st.tabs(["Description", "Datasets","1NN-Classification Accuracy","BOP Classification Accuracy","Tightness of Lower Bound"]) 


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
    st.markdown('# Bag-Of-Patterns Classification Accuracy Results')
    container_accuracy_method = st.container()
    all_method = st.checkbox("Select all",key='all_method')
    if all_method: methods_family = container_accuracy_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
    else: methods_family = container_accuracy_method.multiselect('Select a group of methods',methods, key='selector_methods')

    container_accuracy_metric = st.container()
    all_metric = st.checkbox('Select all',key='all_metrics')
    if all_metric: metrics = container_accuracy_metric.multiselect('Select metric',list_measures,list_measures)
    else: metrics = container_accuracy_metric.multiselect('Select metric',list_measures)

    box_df = generate_dataframe(results,datasets,methods_family,metrics)
    plot_boxplot(box_df,metrics,datasets,methods_family)

with tab_1nn_classification:
    st.markdown('# 1-Nearest Neighbor Classification Accuracy Results')
    container_1nn_accuracy_method = st.container()
    all_onenn_method = st.checkbox("Select all",key='all_onenn_method')
    if all_onenn_method: onenn_methods_family = container_1nn_accuracy_method.multiselect('Select a group of methods', onenn_methods_list, onenn_methods_list, key='selector_onenn_methods_all')
    else: onenn_methods_family = container_1nn_accuracy_method.multiselect('Select a group of methods',onenn_methods_list, key='selector_onenn_methods')

    container_1nn_accuracy_metric = st.container()
    all_onenn_metric = st.checkbox('Select all',key='all_onenn_metrics')
    if all_onenn_metric: onenn_metrics = container_1nn_accuracy_metric.multiselect('Select metric',onenn_metrics_list,onenn_metrics_list)
    else: onenn_metrics = container_1nn_accuracy_metric.multiselect('Select metric',onenn_metrics_list)

    onenn_box_df = generate_dataframe(onenn_results,datasets,onenn_methods_family,onenn_metrics)
    plot_boxplot(onenn_box_df,onenn_metrics,datasets,onenn_methods_family,key='table_onenn')

with tab_tlb:
    st.markdown('# Tightness of Lower Bound Results')
    # container_tlb_method = st.container()
    # all_method = st.checkbox("Select all",key='tlb_method')
    # if all_method: tlb_family = container_tlb_method.selectbox('Select a group of methods', methods, methods, key='selector_methods_all')
    # else: tlb_family = container_tlb_method.selectbox('Select a group of methods',methods, key='selector_tlb_method')

    container_tlb = st.container()
    all_metric = st.checkbox('Select all',key='all_tlbs')
    if all_metric: tlb_dataset = container_tlb.selectbox('Select dataset',sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
    else: tlb_dataset = container_tlb.selectbox('Select dataset',sorted(find_datasets(cluster_size, length_size, types)))

    tlb_results = tlb_dfs[tlb_dataset]
    tlb_results = tlb_results.replace('sax','SAX')
    tlb_results = tlb_results.replace('sfa','SFA')
    tlb_results = tlb_results.replace('spartan','SPARTAN')

    # tlb_results = tlb_results[tlb_results['method'] == tlb_family]
    sax_tlb_values = tlb_results[tlb_results['method']=='SAX'][['tlb']].to_numpy()
    sfa_tlb_values = tlb_results[tlb_results['method']=='SFA'][['tlb']].to_numpy()
    spartan_tlb_values = tlb_results[tlb_results['method']=='SPARTAN'][['tlb']].to_numpy()

    # tlb_x = tlb_results['a']
    # tlb_y = tlb_results['w']
    # tlb_z = tlb_results['tlb']

    fig = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133,projection='3d')

    width = depth = 1


    x,y = np.meshgrid(alphabet_sizes,word_sizes)
    bottom = np.zeros_like(x)

    # print(x)
    # print(y)

    ax1.bar3d(x.ravel(),y.ravel(),bottom.ravel(),width,depth,spartan_tlb_values.ravel(),shade=True)
    ax1.invert_xaxis()
    # ax1.set_zlim(0.7)
    ax1.set_zlim(0,0.7)
    ax1.set_xlabel('w: word_length')
    ax1.set_ylabel('a: alphabet_size')
    ax1.set_zlabel('mean tlb')
    ax1.set_title(f'SPARTAN Mean TLB {tlb_dataset}')

    ax2.bar3d(x.ravel(),y.ravel(),bottom.ravel(),width,depth,sax_tlb_values.ravel(),shade=True)
    ax2.invert_xaxis()
    ax2.set_zlim(0,0.7)
    ax2.set_xlabel('w: word_length')
    ax2.set_ylabel('a: alphabet_size')
    ax2.set_zlabel('mean tlb')
    ax2.set_title(f'SAX Mean TLB {tlb_dataset}')

    ax3.bar3d(x.ravel(),y.ravel(),bottom.ravel(),width,depth,sfa_tlb_values.ravel(),shade=True)
    ax3.invert_xaxis()
    ax3.set_zlim(0,0.7)
    ax3.set_xlabel('w: word_length')
    ax3.set_ylabel('a: alphabet_size')
    ax3.set_zlabel('mean tlb')
    ax3.set_title(f'SFA* Mean TLB {tlb_dataset}')

    # fig = go.Figure()

    # fig.add_trace(plotly_bar_charts_3d(tlb_x,tlb_y,tlb_z,color='x+y', x_title='Alphabet Size', y_title='Word Length',z_title= 'TLB'))

    # fig = plotly_bar_charts_3d(tlb_x,tlb_y,tlb_z,color='x+y', x_title='Alphabet Size', y_title='Word Length',z_title= 'TLB')
    st.pyplot(fig)


