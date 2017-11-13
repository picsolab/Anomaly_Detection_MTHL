# Anomaly_Detection_MTHL

Multi-view Time-series Hypersphere Learning -- MTHL

## Title

Anomaly Detection in Dynamic Networks Using Multi-View Time-Series Hypersphere Learning, CIKM 2017

## Abstract

Detecting anomalous patterns from dynamic and multi-attributed network systems has been a challenging problem due to the complication of temporal dynamics and the variations reflected in multiple data sources. We propose a Multi-view Time-Series Hypersphere Learning (MTHL) approach that leverages multi-view learning and support vector description to tackle this problem. Given a dynamic network with time-varying edge and node properties, MTHL projects multi-view time-series data into a shared latent subspace, and then learns a compact hypersphere surrounding normal samples with soft constraints. The learned hypersphere allows for effectively distinguishing normal and abnormal cases. We further propose an efficient, two-stage alternating optimization algorithm as a solution to the MTHL. Extensive experiments are conducted on both synthetic and real datasets. Results demonstrate that our method outperforms the state-of-the-art baseline methods in detecting three types of events that involve (i) time-varying features alone, (ii) time-aggregated features alone, as well as (iii) both features. Moreover, our approach exhibits consistent and good performance in face of issues including noises, anomaly pollution in training phase and data imbalance.

## Packages

cPickle, numpy, time, sklearn, math

## Parameters

p: the reduced dimensionality for feature
q: the reduced dimensionality for timesteps
lambda1: tradeoff parameter for anomaly pollution in hypersphere learning item
lambda2: tradeoff parameter for the temporal smoothing item
gamma: update step
s: the time window parameter to guarantee temporal smoothing
convergence: convergence tolerance
tolerance: anomaly tolerance, e.g. only if a data point meets d^2 >= r^2*(1+tolerance) we can say it is anomalous

## Usage

"train.pkl" and "test.pkl" are two example datasets, which contains both data and ground truth labels (for evaluation). To run MTHL algorithm on them, please try:

```python main.py```

## Citation

If you make use of this code, please kindly cite our paper:
Teng, X., Lin, Y.-R., Wen, X. (2017). "Anomaly Detection in Dynamic Networks using Multi-view Time-Series Hypersphere Learning." In Proceedings of  the 26th ACM International Conference on Information and Knowledge Management (CIKM 2017) https://doi.org/10.1145/3132847.3132964
