import pandas as pd
import numpy as np




def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd

methods = ['SAX','SFA','SPARTAN']
onenn_methods_list =['SAX','SFA','SPARTAN','SAX-DR','SAX_VFD','TFSAX','1d-SAX','ESAX']

classification_types = ['1NN','BOP']

bop_metrics_list = ['Euclid','BOSS','Cosine','KL-Div']
onenn_metrics_list = ['symbolic-l1']

runtime_options = ['Mean Accuracy', 'Mean Rank']

word_sizes = np.arange(2,9,1)
alphabet_sizes = np.arange(3,11,1)

significance_optons = ['0.1','0.05']
stat_test_options = ['nemenyi','bonferroni-dunn']



def find_datasets(clusters_size, lengths_size, types):
    df = pd.read_csv('data/characteristics.csv')
    def f(row):
        if row['SeqLength'] < 100:
            val = 'VERY-SMALL(<100)'
        elif row['SeqLength'] < 300:
            val = 'SMALL(<300)'
        elif row['SeqLength'] < 500:
            val = 'MEDIUM(<500)'
        elif row['SeqLength'] < 1000:
            val = 'LARGE(<1000)'
        else:
            val = 'VERY-LARGE(>1000)'
        return val

    df['LengthLabel'] = df.apply(f, axis=1)

    def f(row):
        if row['NumClasses'] < 10:
            val = 'VERY-SMALL(<10)'
        elif row['NumClasses'] < 20:
            val = 'SMALL(<20)'
        elif row['NumClasses'] < 40:
            val = 'MEDIUM(<40)'
        else:
            val = 'LARGE(>40)'
        return val

    df['ClustersLabel'] = df.apply(f, axis=1)

    if len(clusters_size) > 0:
        df = df.loc[df['ClustersLabel'].isin(clusters_size)] 
    if len(lengths_size) > 0:
        df = df.loc[df['LengthLabel'].isin(lengths_size)]
    if len(types) > 0:
        df = df.loc[df['TYPE'].isin(types)]

    return list(df['Name'].values)



list_type = ['AUDIO','DEVICE','ECG','EOG','EPG','HEMODYNAMICS','IMAGE','MOTION','OTHER','SENSOR','SIMULATED','SOUND','SPECTRO','TRAFFIC']

list_seq_length = ['VERY-SMALL(<100)', 'SMALL(<300)', 'MEDIUM(<500)', 'LARGE(<1000)', 'VERY-LARGE(>1000)']

list_num_clusters = ['VERY-SMALL(<10)', 'SMALL(<20)', 'MEDIUM(<40)', 'LARGE(>40)']

list_measures = ['symbolic-l1','Euclid','BOSS','Cosine','KL-Div']

list_length = [16,32,64,128,256,512,768,1024]



description_intro1 = f"""
Symbolic Representation, a dimensionality reduction technique that transforms time series into a string of discrete symbols, has received increasing attention in downstream tasks such as similarity search, indexing, anomaly detection, and classification.
Despite vast progress in this area, the majority of solutions still rely on a well-established method proposed two decades ago for its simplicity, despite other potential solutions with stronger representation power.
Moreover, from the existing literature, there is a noticeable absence of a comprehensive study in this domain, highlighting a need for more in-depth investigation.
Motivated by the aforementioned issues, we propose SAIL, a modular web engine serving two purposes: (i) to provide the first comprehensive evaluation studies on 7 state-of-the-art symbolic representation methods over 128 time-series datasets, the largest study in this area;
(ii) to demonstrate the superiority of a recently proposed solution, SPARTAN, that mitigates the non-negligible drawbacks of the uniform balance in symbolic subspaces, which assumes a single alphabet size for each symbol.
Through the interactive interface, SAIL facilitates users to explore, visualize, and comprehend the quantitative assessment results on various methods, datasets, and metrics.
From diverse scenarios, SPARTAN has demonstrated superior performance across different dimensions.
Overall, SAIL not only facilitates the most comprehensive evaluation study in this field, but also delivers new insights and concrete solutions, laying the groundwork for the next-generation symbolic representation solutions.
"""


description_intro2 = f"""

## User Guide

This demonstration is organized as follows:
- Datasets: An overview of the datasets used for comparison, sourced from the UCR Time Series Classification Archive [1], and their relevant sizes and data types. These datasets can be selected based on these characteristics using the sidebar to carry out analysis of the performance on specific sizes or types of data.
- Methods: A review of the methods shown in this paper, along with summaries of their symbolic representation process and links to the source papers.
- 1NN-Classification Accuracy: A review of the classification accuracy of equal sized representations for each symbolic reprsentation method using an equivalent distance metric for all methods. Presents average accuracy, pairwise, and ranked statistical comparisons.
- BOP Classifcation Accuracy: A review of the important Bag-Of-Patterns classification setting with an extensive list of distance measures available for comparison.
- Tightness of Lower Bound: An important characteristic of a symbolic representation method is the ability to lower bound the euclidean distance, a useful property for indexing applications, this page provides a tool for examining the scaling power of symbolic representations for producing tight lower bounds and the statistical tests to show superior performers under different settings.
- Runtime Analysis: An examination of the runtime/accuracy tradeoff of the sampled symbolic reprsentation methods, presented both in terms of mean accuracy and rank over the chosen datasets and including the views on both the training and inference time requirements.

## Contributors

* [Ryan DeMilt*](https://github.com/ryanpdemilt) (Ohio State University) (demilt.4@osu.edu)
* [Fan Yang*](https://github.com/Ivan-Fan) (Ohio State University) (yang.7007@osu.edu)
* [John Paparrizos](https://www.paparrizos.org/) (Ohio State University) (paparrizos.1@osu.edu)


## Datasets

To ease reproducibility, we share our results over an established benchmark:

* The UCR Univariate Archive, which contains 128 univariate time-series datasets.
    * Download all 128 preprocessed datasets [here](https://www.thedatum.org/datasets/UCR2022_DATASETS.zip).

For the preprocessing steps check [here](https://github.com/thedatumorg/UCRArchiveFixes).

## Models

We evaluate using a broad survey of time series symbolic representation methodologies. We separate these into three categories. The Symbolic Aggregate Approximation (SAX) and variant methods, which discretize the time series by splitting the series into segments and assign symbols to each second based on summary statistics such as the mean, min, max, a combination. Some of the methods introduce bespoke features to summarize the series segment such as trend or direction features.
The second type makes use of the Discrete Fourier Transform to summarize entire time series with some loss of high frequency information. This method, the Symbolic Fourier Analysis (SFA) has been broadly applied and seen wide adoption in dictionary classification methods since its release, being used as the central component of classifiers such as BOSS, WEASEL, and TDE among others.
The final representationt type we explore is the newly introduced Symbolic PCA Representation for Time Series ANalytics (SPARTAN). SPARTAN exploits information from the entire dataset using well known dimensionality reduction techniques and utilizes optimized allocation of storage resources to make superior use of the representation space available to the model. These improvements lead to SPARTAN achieving 2X better tightness of lower bounding and statistically significant improvements in classification downstream tasks. SPARTAN also improves by 2X inference runtime performance compared to the current best solutions.
We use this set of methods to comprehensively survey the space of time series classification methods, and provide a web-based interface to explore the relative performance of these tools.

"""

text_description_dataset = f"""
We conduct our evaluation using the UCR Time-Series Archive, the largest collection of class-labeled time series datasets. 
The archive consists of a collection of 128 datasets sourced from different sensor readings while performing diverse tasks from multiple 
domains. All datasets in the archive span between 40 to 24000 time-series and have lengths varying from 15 to 2844. Datasets are z-normalized, 
and each time-series in the dataset belongs to only one class. There is a small subset of datasets in the archive containing missing values and 
varying lengths. We employ linear interpolation to fill the missing values and resample shorter time series to reach the longest time series 
in each dataset. Here we present the characteritics of each dataset in the UCR archive.
"""

text_description_models = f"""
We present the models which are being compared here, each with a short description of their methodology and a citation to the work in which they were introduced.

| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Symbolic Representation </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Description </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:------------------------------------|
||  **SAX and SAX-variant methods**
|SAX             | SAX builds on Piecewise Aggregate Approximation to approximate a time series using the means of equal-length, non-overlapping segments of the time series. SAX converts these approximations to symbols by dividing the range of possible values in the number line according to equi-probability sections of a normal distribution with mean zero and unit variance.|                 [2]          |
|ESAX              | ESAX uses a similar methodology to SAX but extracts two additional features from each segment: the minimum and maximum value of the segments. This is done in order to capture more information from the segments than the mean, though this triples the storage requiremnts for an equal number of segments between methods.                                |          [3]                 |
|TSFSAX              |  TSFSAX introduces a novel trend feature based on trend points in the segments which is later discretized along with the mean.                           |            [4]               |
|SAX-DR             | SAX-DR derives an additional categorization of the trend of each segment, classifying each as concave, convex, or linear in order to encode local shape of the segments.                              |           [5]                |
|SAX-VFD          |SAX-VFD adopts 18 features from three categories: statistical, entropy and fluctuation features. An optimization process is then applied to determine an optimal feature fector to be used from among these.                         |           [6]                 |
|1D-SAX          |1d-SAX extends the classic SAX feature with a new slope feature extracted using linear regression on each segment.|       [7]                     |
| |**DFT-Based Representation Method**
|SFA           | SFA, leverages Discrete Fourier Transform (DFT) to capture the frequency information from the Fourier domain. In addition, Multiple Coefficient Binning (MCB) is also proposed as the standard discretization technique for alphabet dictionaries                                |        [8]                    |
|  | **Newly Proposed Symbolic Representation Method**
|SPARTAN              | SPARTAN utilizes a data adaptive dimensionality reduction method to extract information from the entire available training set. This method discretizes the space in an uneven manner utilizing the importance of each subspace in the approximation step to guide the size of the alphabet allocated to it. Through this SPARTAN is able to achieve a rich and informative symbolic representation with knowledge extracted from trends across the entire dataset and budgeted by importance of the sampled subspaces.                               |         [9]                   |
"""

text_1nn_classification_description = f"""
In this section we present the results for the 1-Nearest Neighbor classification task. In this experiment all methods are given an equal size representation space of 12 symbols per word and an alphabet size of 4 (since SPRATAN dynamically calculates alphabets it is given a bit budget of 24 for this experiment to give an equal number of possible generated words). Each time series is converted to symbolic representations of the appropriate size and classification is carried out by identifying the nearest neighbor of a queried time series in the training set and assigning the same label. This requires a meaningful distance measure over symbolic representations, for fairness it is ideal to use the same distance measure across all representation methods so that no method is unfairly advantaged by a measure which makes use of internal information prior to the symbolic representation step such as histogram breakpoints. To perform this evaluation we suggest a simple but effective metric which takes the L1-distance with respect to the ordered alphabets of symbols, which we reproduce as follows.
"""

test_symbolic_l1_distance = f"""
    d_{{l1}}(\\tilde{{Q}},\\tilde{{C}}) = \\sum_{{i=1}}^{{\\omega}} \\lvert i(\\tilde{{Q}}_i) - i(\\tilde{{C}}_i) \\rvert \\text{{where,}} i(\\tilde{{Q}}_i):\\Phi \\rightarrow \\{{1,\\ldots,|\\Phi|-1\\}} \\\\ \\text{{ and where }} \\omega \\text{{ is the word length, }} \\alpha \\text{{ is the alphabet size,}} \\\\ \\text{{and the alphabet }} \\Phi \\text{{ can be naturally ordered and assigned to an index with natural numbers via function: }} i(\\cdot).
"""
# text_symbolic_l1_distance_description = f"""
#     \\text{{where }} \\omega \\text{{ is the word length, }} \\alpha \\text{{is the alphabet size, and the alphabet }} \\Phi \\text{{ can be naturally ordered and assigned to an index with natural numbers.}}
# """
text_1nn_classification_description_2 = f"""
    We provide functionality to view the classification results through multiple views, each explained with text on the corresponding view, along with analysis of the results.
"""

text_boxplot_explanation = f"""
This panel presents a boxplot displaying the average classification accuracy for this task and allows the user to select which methods they would like to examine the accuracy for, the second selection box allows the user to choose from among the available metrics for the classification type being displayed. Below this plot the user can find a table of the raw classification accuracies for each method and each dataset for a full comparison.
"""

text_pairwise_comparison = f"""
On this panel we present pairwise accuracy comparisons, the user can select combinations of symbolic representation method and distance metric, results above the diagonal line are datasets where method 2 exhibits superior classification accuracy to method 1 and the reverse for below the diagonal line.
"""

text_cd_diagram_explanation = f"""
Here we present the the critical difference diagram for our results, introduced by [10]. This diagram aims to rigourously evaluate the performance of multiple classifiers over multiple datasets. Since the individual classification accuracies may not be commensurably comparable, we instead present the ranking of each method on that dataset relative to the other methods (i.e. the best performing method receives rank 1, the second best rank 2). The CD diagram shows each method with a marker under its average ranking over all 128 datasets. A post-hoc test is then applied to find cliques of methods whose performance is not statistically significantly different. A clique of methods is shown by a horizontal line between two methods. If two methods have no line between them then these methods have been observed to have a statistically significant difference in accuracy based on the sample of classification accuracies from the 128 datasets. Here we allow for the presentation of two methods for clique finding, the Nemenyi and Bonferroni-Dunn tests. We also have available two common significance levels to display cliques among the data. We also note that in the community it is generally noted that to reasonably observe significance between methods, one needs ~10 datasets per method under observation ideally. For this reason it is recommended under these conditions to perform explanations of twelve or fewer methods, given there are 128 observed datasets. 
"""

text_bop_classification_description = f"""
The second classification task we present is the Bag-Of-Words task. In this task we first divide the input time series dataset into sliding, overlapping windows of length 12. To each time series we apply a symbolic representation method and assign a word of length 4 and alphabet size 4 (dynamically allocated with a bit budget of 8 in the case of SPARTAN). The collection of words generated for each time series forms a histogram which can be viewed as the occurences of a given pattern in that time series. This allows the symbolic representation method to serve as a feature extractor to derive important patterns from the input time series. For the BOP task we perform 1-NN classification using these histograms. Here it is important to have a distance metric which effectively captures the distance between these histograms which when normalized can be considered as probability distributions. Many distance metrics exist for this comparison and there is a rich literature of work on comparing probability measures. We present here the Euclidean distance between histograms, the BOSS distance [11], the Cosine-similarity, and the KL-Divergence.
"""

text_tlb_description = f"""
We show here the comparison of SAX, SFA, and SPARTAN on lower bounding the euclidean distance, when equipped with a function MINDIST which is guaranteed to lower bound the euclidean distance. We note that SFA does not include a lower bouding distance between two symbolic representations in its work, presenting a lower bound between a non discretized DFT and an SFA representation. Due to this we show TLB results for SFA using a modified MINDIST function, this function holds as a lower bound for the euclidean distance in most cases but does not have a corresponding proof. Therefore the primary comparison is with SAX in this case and SFA is included for completeness. WE present comparative plots for the tightness of lower bound as the word length and alphabet size parameters scale for each method. We also present a critical diagram to show a statistical comparison of the tightness of lower bounding capabilities between methods.
"""

text_runtime_description = f"""
This page investigates the runtime, accuracy tradeoff between methods. To evaluate the runtime of methods we measure the time number of seconds per time series to transform each dataset in the UCR archive. To further illuminate the differences between methods we present runtime in terms of total runtime (training and inference), as well as training and inference runtime separately. In doing so we highlight that the runtime/accuracy tradeoff can have significant differences between training and testing phases, and these differences have large relevance for application settings of symbolic representations. We present SPARTAN with and without the speedup that can be achieved through a randomized SVD solver [10] for clarity.
"""
references = f"""
[1] Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah
Mueen and Gustavo Batista (2015). The UCR Time Series Classification Archive. URL
www.cs.ucr.edu/~eamonn/time_series_data/

[2] Jessica Lin, Eamonn Keogh, Stefano Lonardi, and Bill Chiu. 2003. A symbolic
representation of time series, with implications for streaming algorithms. In
Proceedings of the 8th ACM SIGMOD workshop on Research issues in data mining
and knowledge discovery. 2–11.

[3] Battuguldur Lkhagva, Yu Suzuki, and Kyoji Kawagoe. 2006. Extended SAX:
Extension of symbolic aggregate approximation for financial time series data
representation. DEWS2006 4A-i8 7 (2006).

[4] Tianyu Li, Fang-Yan Dong, and Kaoru Hirota. 2013. Distance Measure for Sym-
bolic Approximation Representation with Subsequence Direction for Time Series
Data Mining. Journal of Advanced Computational Intelligence and Intelligent
Informatics 17, 2 (2013), 263–271

[5] Yufeng Yu, Yuelong Zhu, Dingsheng Wan, Qun Zhao, and Huan Liu. 2019. A
novel trend symbolic aggregate approximation for time series. arXiv preprint
arXiv:1905.00421 (2019).

[6] Lijuan Yan, Xiaotao Wu, and Jiaqing Xiao. 2022. An Improved Time Series
Symbolic Representation Based on Multiple Features and Vector Frequency
Difference. Journal of Computer and Communications 10, 06 (2022), 44–62.

[7] Simon Malinowski, Thomas Guyet, René Quiniou, and Romain Tavenard. 2013.
1d-sax: A novel symbolic representation for time series. In International Sympo-
sium on Intelligent Data Analysis. Springer, 273–284.

[8] Patrick Schäfer and Mikael Högqvist. 2012. SFA: a symbolic fourier approxima-
tion and index for similarity search in high dimensional datasets. In Proceedings
of the 15th International Conference on Extending Database Technology (Berlin,
Germany) (EDBT ’12). Association for Computing Machinery, New York, NY,
USA, 516–527. https://doi.org/10.1145/2247596.224765

[9] Ryan DeMilt, Fan Yang, and John Paparrizos. 2024. SPARTAN: Data-adaptive
Symbolic Representations for Time Series Data Analysis. (2024). Under review

[10] Janez Demšar. 2006. Statistical Comparisons of Classifiers over Multiple Data
Sets. J. Mach. Learn. Res. 7 (dec 2006), 1–30.

[11] Patrick Schäfer. 2015. The BOSS is concerned with time series classification
in the presence of noise. Data Min. Knowl. Discov. 29, 6 (nov 2015), 1505–1530.
https://doi.org/10.1007/s10618-014-0377-7
"""