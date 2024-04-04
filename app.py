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
import plotly.express as px

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

onenn_results['method_metric'] = onenn_results['method'] + '+' + 'symbolic-l1'

onenn_results = pd.pivot(onenn_results,index='dataset',columns = 'method_metric',values='acc')
onenn_results= onenn_results.reset_index()

onenn_methods_list =['SAX','SFA','SPARTAN','SAX-DR','SAX_VFD','TFSAX','1d-SAX','ESAX']

acc_results = pd.read_csv('./data/symbol_results.csv')
runtime_results = pd.read_csv('./data/runtime_results_randomized.csv')

acc_results = acc_results[acc_results['method'] != 'spartan_pca']

acc_results = acc_results.replace('sax','SAX')
acc_results = acc_results.replace('sfa','SFA')
# acc_results = acc_results.replace('spartan_pca','SPARTAN')
acc_results = acc_results.replace('spartan_pca_allocation','SPARTAN')

acc_results = acc_results.set_index('method')

scaling_ranks = acc_results.groupby(by=['dataset'])['acc'].rank(ascending=False)
acc_results['rank'] = scaling_ranks
acc_results = acc_results.reset_index()


# acc_results['SPARTAN+speedup'] = acc_results['SPARTAN']

runtime_results = runtime_results.replace('PAA','SAX')
runtime_results = runtime_results.replace('DFT','SFA')
runtime_results = runtime_results.replace('PCA','SPARTAN')
runtime_results = runtime_results.replace('PCA+Allocation','SPARTAN')
runtime_results = runtime_results.replace('PCA+Allocation_randomized','SPARTAN+speedup')

spartan_results = acc_results[acc_results['method'] == 'SPARTAN']
spartan_results['method'] = 'SPARTAN+speedup'

acc_results = pd.concat([acc_results,spartan_results])

classification_types = ['1NN','BOP']
# full_results = acc_results.join(runtime_results,on=['method','dataset'],)

# runtime_results = pd.merge(acc_results,runtime_results,how='left',on=['method','dataset'])
# runtime_results['train_time'] = runtime_results['train_time']*1000
# runtime_results['pred_time'] = runtime_results['pred_time']*1000



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


    st.markdown('# Classification Accuracy Per Dataset')
    cols_list = []
    for i, col in enumerate(df.columns):
        if i > 0:
            cols_list.append(col)
        else:
            cols_list.append(col)

    df.columns = cols_list
    AgGrid(df,key=key,reload_data=True,fit_columns_on_grid_load=True)
def plot_stat_plot(df, datasets,stat_methods_family,metrics,classification_type='1nn'):
    # container_method = st.container()
    # stat_methods_family = container_method.multiselect('Select a group of methods', sorted(methods_family), key='selector_stat_methods')
    
    # df = df.loc[df['dataset'].isin(datasets)][[method_g + '+' + metric for metric in metrics for method_g in stat_methods_family]]
    # df.insert(0, 'dataset', datasets)

    # [method_g + '+' + metric for metric in metrics for method_g in stat_methods_family]
    significance_optons = ['0.1','0.05']
    stat_test_options = ['nemenyi','bonferroni-dunn']

    container_stat_test = st.container()
    stat_test = container_stat_test.selectbox('Select Statistical Test',stat_test_options,index=0,key='stat_test_select_' + classification_type)
    significance = container_stat_test.selectbox('Select Significance Level',significance_optons,index=0,key='significance_level_select_' + classification_type)


    if len(datasets) > 0:
        if len(stat_methods_family) > 1 and len(stat_methods_family) < 13:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df

                df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method)

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, alpha=significance,test=stat_test)
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
                rank_df = rank_df.reset_index()
                rank_df.columns = ['Method Name', 'Average Rank']
                st.table(rank_df)

            stat_plots(df)


with st.sidebar:
    st.markdown('# SAIL: Symbolic Representation Explorer')
     
    # container_metric = st.container()
    # all_metric = st.checkbox('Select all',key='all_metrics')
    # if all_metric: metrics = container_metric.multiselect('Select metric',list_measures,list_measures)
    # else: metrics = container_metric.multiselect('Select metric',list_measures)
    
    container_dataset = st.container()  
    all_cluster = st.checkbox("Select all", key='all_clusters',value=True)
    if all_cluster: cluster_size = container_dataset.multiselect('Select number of classes', sorted(list(list_num_clusters)), sorted(list(list_num_clusters)))
    else: cluster_size = container_dataset.multiselect('Select number of classes', sorted(list(list_num_clusters)))

    container_dataset = st.container()  
    all_length = st.checkbox("Select all", key='all_lengths',value=True)
    if all_length: length_size = container_dataset.multiselect('Select sequence length size', sorted(list(list_seq_length)), sorted(list(list_seq_length)))
    else: length_size = container_dataset.multiselect('Select length sequence size', sorted(list(list_seq_length)))

    container_dataset = st.container()  
    all_type = st.checkbox("Select all", key='all_types',value=True)
    if all_type: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)), sorted(list(list_type)))
    else: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)))
   
    container_dataset = st.container()  
    all_dataset = st.checkbox("Select all", key='all_dataset',value=True)
    if all_dataset: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
    else: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types)))

    # container_method = st.container()
    # all_method = st.checkbox("Select all",key='all_method')
    # if all_method: methods_family = container_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
    # else: methods_family = container_method.multiselect('Select a group of methods',methods, key='selector_methods')

# tab_desc, tab_acc, tab_time, tab_stats, tab_analysis, tab_misconceptions, tab_ablation, tab_dataset, tab_method = st.tabs(["Description", "Evaluation", "Runtime", "Statistical Tests", "Comparative Analysis", "Misconceptions", "DNN Ablation Analysis", "Datasets", "Methods"]) 
tab_desc, tab_dataset,tab_1nn_classification,tab_classification_accuracy,tab_tlb,tab_runtime = st.tabs(["Description", "Datasets","1NN-Classification Accuracy","BOP Classification Accuracy","Tightness of Lower Bound","Runtime Analysis"]) 


with tab_desc:
    st.markdown('# SAIL: A Voyage to Symbolic Representation Solutions for Time-Series Analysis')
    st.markdown(description_intro1)
    background = Image.open('./data/spartan_demo_pipeline.png')
    col1, col2, col3 = st.columns([1.2, 5, 0.2])
    col2.image(background, width=900, caption='Overview of the SAIL representation method.')
    # st.markdown(description_intro2)
    # background = Image.open('./data/taxonomy.png')
    # col1, col2, col3 = st.columns([1.2, 5, 0.2])
    # col2.image(background, width=900, caption='Taxonomy of time-series clustering methods in Odyssey.')

with tab_dataset:
    st.markdown('# Dataset Description')
    st.markdown(text_description_dataset)
    AgGrid(characteristics_df)

with tab_classification_accuracy:

    tab_bop_boxplot,tab_bop_pairwise,tab_bop_stats = st.tabs(['Boxplot','Pairwise','Statistical Tests'])

    with tab_bop_boxplot:
        st.markdown('# Bag-Of-Patterns Classification Accuracy Results')
        container_accuracy_method = st.container()
        all_method = st.checkbox("Select all",key='all_method',value=True)
        if all_method: methods_family = container_accuracy_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
        else: methods_family = container_accuracy_method.multiselect('Select a group of methods',methods, key='selector_methods')

        container_accuracy_metric = st.container()
        all_metric = st.checkbox('Select all',key='all_metrics',value=True)
        if all_metric: metrics = container_accuracy_metric.multiselect('Select metric',list_measures,list_measures)
        else: metrics = container_accuracy_metric.multiselect('Select metric',list_measures)

        box_df = generate_dataframe(results,datasets,methods_family,metrics)
        plot_boxplot(box_df,metrics,datasets,methods_family)
    with tab_bop_pairwise:
        option1 = st.selectbox('Method 1',tuple(methods),index=0)
        metric1 = st.selectbox('Metric 1',bop_metrics_list,index=0)
        # methods_family = methods_family[1:] + methods_family[:1]
        option2 = st.selectbox('Method 2',tuple(methods),index=0)
        metric2 = st.selectbox('Metric 2',bop_metrics_list,index=0)

        method_metric_1 = option1 + '+' + metric1
        method_metric_2 = option2 + '+' + metric2

        if len(methods_family) > 0 and len(datasets) > 0:
            fig = go.FigureWidget()
            trace1 = fig.add_scattergl(x=box_df[method_metric_1], y=box_df[method_metric_2], mode='markers', name='(' + method_metric_1 + '+'+ method_metric_2 +')',  text=datasets,
                                    textposition="bottom center",
                                    marker = dict(size=10,
                                                opacity=.7,
                                                color='red',
                                                line = dict(width=1, color = '#1f77b4')
                                                ))
            fig.add_trace(go.Scatter(
                                x=[min(min(box_df[method_metric_1])+1e-4, min(box_df[method_metric_2])+1e-4), max(max(box_df[method_metric_1])+1e-4, max(box_df[method_metric_2])+1e-4)],
                                y=[min(min(box_df[method_metric_1])+1e-4, min(box_df[method_metric_2])+1e-4), max(max(box_df[method_metric_1])+1e-4, max(box_df[method_metric_2])+1e-4)],
                                name="X=Y"
                            ))
            trace2 = fig.add_histogram(x=box_df[method_metric_1], name='x density', marker=dict(color='#1f77b4', opacity=0.7),
                                yaxis='y2'
                                )
            trace3 = fig.add_histogram(y=box_df[method_metric_2], name='y density', marker=dict(color='#1f77b4', opacity=0.7), 
                                xaxis='x2'
                                )
            fig.layout = dict(xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            xaxis_title=option1, yaxis_title=option2,
                            showlegend=False,
                            margin=dict(t=50),
                            hovermode='closest',
                            bargap=0,
                            xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            height=600,
                            )

            def do_zoom(layout, xaxis_range, yaxis_range):
                inds = ((xaxis_range[0] <= box_df[method_metric_1]) & (box_df[method_metric_1] <= xaxis_range[1]) &
                        (yaxis_range[0] <= box_df[method_metric_2]) & (box_df[method_metric_2] <= yaxis_range[1]))

                with fig.batch_update():
                    trace2.x = box_df[method_metric_1][inds]
                    trace3.y = box_df[method_metric_2][inds]
                
            fig.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')
            fig.update_xaxes(tickfont_size=16)
            fig.update_yaxes(tickfont_size=16)
            st.plotly_chart(fig)
    with tab_bop_stats:
        metric_options = bop_metrics_list
        cd_df = results
        methods_list = methods

        container_cd = st.container()
        all_cd_metrics = st.checkbox('Select all',key='all_cd_metrics_bop',value=True)
        if all_cd_metrics: cd_metric = container_cd.multiselect('Select metric',metric_options,metric_options,key='selector_cd_metrics_all_bop')
        else: cd_metric = container_cd.multiselect('Select metric',metric_options,key='selector_cd_metrics_bop')

        container_cd_accuracy_method = st.container()
        all_cd_method = st.checkbox("Select all",key='all_cd_method_bop',value=True)
        if all_cd_method: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods', methods_list, methods_list, key='selector_cd_methods_all_bop')
        else: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods',methods_list, key='selector_cd_methods_bop')

        cd_df_subset = generate_dataframe(cd_df,datasets,cd_methods_family,cd_metric)
        plot_stat_plot(cd_df_subset,datasets,cd_methods_family,cd_metric,'bop')

with tab_1nn_classification:

    tab_1nn_boxplot,tab_1nn_pairwise,tab_1nn_stats = st.tabs(['Boxplot','Pairwise','Statistical Tests'])
    with tab_1nn_boxplot:
        st.markdown('# 1-Nearest Neighbor Classification Accuracy Results')
        container_1nn_accuracy_method = st.container()
        all_onenn_method = st.checkbox("Select all",key='all_onenn_method',value=True)
        if all_onenn_method: onenn_methods_family = container_1nn_accuracy_method.multiselect('Select a group of methods', onenn_methods_list, onenn_methods_list, key='selector_onenn_methods_all')
        else: onenn_methods_family = container_1nn_accuracy_method.multiselect('Select a group of methods',onenn_methods_list, key='selector_onenn_methods')

        container_1nn_accuracy_metric = st.container()
        all_onenn_metric = st.checkbox('Select all',key='all_onenn_metrics',value=True)
        if all_onenn_metric: onenn_metrics = container_1nn_accuracy_metric.multiselect('Select metric',onenn_metrics_list,onenn_metrics_list)
        else: onenn_metrics = container_1nn_accuracy_metric.multiselect('Select metric',onenn_metrics_list)

        onenn_box_df = generate_dataframe(onenn_results,datasets,onenn_methods_family,onenn_metrics)
        plot_boxplot(onenn_box_df,onenn_metrics,datasets,onenn_methods_family,key='table_onenn')
    with tab_1nn_pairwise:
        option1 = st.selectbox('Method 1',onenn_methods_list,index=0)
        # methods_family = methods_family[1:] + methods_family[:1]
        option2 = st.selectbox('Method 2',onenn_methods_list,index=0)

        method_metric_1 = option1 + '+' + 'symbolic-l1'
        method_metric_2 = option2 + '+' + 'symbolic-l1'

        if len(methods_family) > 0 and len(datasets) > 0:
            fig = go.FigureWidget()
            trace1 = fig.add_scattergl(x=onenn_box_df[method_metric_1], y=onenn_box_df[method_metric_2], mode='markers', name='(' + method_metric_1 + '+'+ method_metric_2 +')',  text=datasets,
                                    textposition="bottom center",
                                    marker = dict(size=10,
                                                opacity=.7,
                                                color='red',
                                                line = dict(width=1, color = '#1f77b4')
                                                ))
            fig.add_trace(go.Scatter(
                                x=[min(min(onenn_box_df[method_metric_1])+1e-4, min(onenn_box_df[method_metric_2])+1e-4), max(max(onenn_box_df[method_metric_1])+1e-4, max(onenn_box_df[method_metric_2])+1e-4)],
                                y=[min(min(onenn_box_df[method_metric_1])+1e-4, min(onenn_box_df[method_metric_2])+1e-4), max(max(onenn_box_df[method_metric_1])+1e-4, max(onenn_box_df[method_metric_2])+1e-4)],
                                name="X=Y"
                            ))
            trace2 = fig.add_histogram(x=onenn_box_df[method_metric_1], name='x density', marker=dict(color='#1f77b4', opacity=0.7),
                                yaxis='y2'
                                )
            trace3 = fig.add_histogram(y=onenn_box_df[method_metric_2], name='y density', marker=dict(color='#1f77b4', opacity=0.7), 
                                xaxis='x2'
                                )
            fig.layout = dict(xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            xaxis_title=option1, yaxis_title=option2,
                            showlegend=False,
                            margin=dict(t=50),
                            hovermode='closest',
                            bargap=0,
                            xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            height=600,
                            )

            def do_zoom(layout, xaxis_range, yaxis_range):
                inds = ((xaxis_range[0] <= box_df[method_metric_1]) & (box_df[method_metric_1] <= xaxis_range[1]) &
                        (yaxis_range[0] <= box_df[method_metric_2]) & (box_df[method_metric_2] <= yaxis_range[1]))

                with fig.batch_update():
                    trace2.x = onenn_box_df[method_metric_1][inds]
                    trace3.y = onenn_box_df[method_metric_2][inds]
                
            fig.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')
            fig.update_xaxes(tickfont_size=16)
            fig.update_yaxes(tickfont_size=16)
            st.plotly_chart(fig)
    with tab_1nn_stats:
        metric_options = onenn_metrics_list
        cd_df = onenn_results
        methods_list = onenn_methods_list

        container_cd = st.container()
        all_cd_metrics = st.checkbox('Select all',key='all_cd_metrics_1nn',value=True)
        if all_cd_metrics: cd_metric = container_cd.multiselect('Select metric',metric_options,metric_options,key='selector_cd_metrics_all_1nn')
        else: cd_metric = container_cd.multiselect('Select metric',metric_options,key='selector_cd_metrics_1nn')

        container_cd_accuracy_method = st.container()
        all_cd_method = st.checkbox("Select all",key='all_cd_method_1nn',value=True)
        if all_cd_method: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods', methods_list, methods_list, key='selector_cd_methods_all_1nn')
        else: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods',methods_list, key='selector_cd_methods_1nn')

        cd_df_subset = generate_dataframe(cd_df,datasets,cd_methods_family,cd_metric)
        plot_stat_plot(cd_df_subset,datasets,cd_methods_family,cd_metric,'1nn')
# with tab_critical_diagrams:
#     st.markdown('# Critical Difference Diagrams for Symbolic Representation Methods')

#     container_type = st.container()
#     typ = container_type.selectbox('Select classification type',classification_types,index=0)

#     metric_options = onenn_metrics_list
#     cd_df = onenn_results
#     methods_list = onenn_methods_list

#     if typ == '1NN':
#         metric_options = onenn_metrics_list
#         cd_df = onenn_results
#         methods_list = onenn_methods_list
#     elif typ =='BOP':
#         metric_options = bop_metrics_list
#         cd_df = results
#         methods_list = methods

#     container_cd = st.container()
#     all_cd_metrics = st.checkbox('Select all',key='all_cd_metrics',value=True)
#     if all_cd_metrics: cd_metric = container_cd.multiselect('Select metric',metric_options,metric_options,key='selector_cd_metrics_all')
#     else: cd_metric = container_cd.multiselect('Select metric',metric_options,key='selector_cd_metrics')

#     container_cd_accuracy_method = st.container()
#     all_cd_method = st.checkbox("Select all",key='all_cd_method',value=True)
#     if all_cd_method: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods', methods_list, methods_list, key='selector_cd_methods_all')
#     else: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods',methods_list, key='selector_cd_methods')

#     cd_df_subset = generate_dataframe(cd_df,datasets,cd_methods_family,cd_metric)
#     plot_stat_plot(cd_df_subset,datasets,cd_methods_family,cd_metric)
with tab_tlb:
    st.markdown('# Tightness of Lower Bound Results')
    # container_tlb_method = st.container()
    # all_method = st.checkbox("Select all",key='tlb_method')
    # if all_method: tlb_family = container_tlb_method.selectbox('Select a group of methods', methods, methods, key='selector_methods_all')
    # else: tlb_family = container_tlb_method.selectbox('Select a group of methods',methods, key='selector_tlb_method')

    container_tlb = st.container()
    # all_metric = st.checkbox('Select all',key='all_tlbs')
    # if all_metric: tlb_dataset = container_tlb.multiselect('Select dataset',sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
    tlb_dataset = container_tlb.selectbox('Select dataset',sorted(find_datasets(cluster_size, length_size, types)),index=0)

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

with tab_runtime:
    st.markdown('# Runtime Analysis')

    acc_results_subset = acc_results[acc_results['dataset'].isin(datasets)]
    runtime_results_subset = runtime_results[runtime_results['dataset'].isin(datasets)]
    scaling_ranks = acc_results_subset.groupby(by=['dataset'])['acc'].rank(ascending=False)
    acc_results_subset['rank'] = scaling_ranks
    acc_results_subset = acc_results.reset_index()

    runtime_results_subset = pd.merge(acc_results_subset,runtime_results_subset,how='left',on=['method','dataset'])
    runtime_results_subset['train_time'] = runtime_results_subset['train_time']*1000
    runtime_results_subset['pred_time'] = runtime_results_subset['pred_time']*1000
    runtime_results_subset['total_time'] = runtime_results_subset['train_time'] + runtime_results_subset['pred_time']

    runtime_results_subset = runtime_results_subset.rename(columns={'total_time':'Runtime','acc':'Mean Accuracy'})
    fig = px.scatter(runtime_results_subset,x='Runtime',y='Mean Accuracy',color='method',log_x=True)

    st.plotly_chart(fig)