# MuGIL
MuGIL: A Multi-Graph Interaction Learning Network for Multi-Task Traffic Prediction

* A multi-graph interaction learning network is proposed for enhancing traffic prediction accuracy across heterogeneous variables and regions.
  
* A multi-source graph representation module is introduced for aligning heterogeneous information through semantic graphs.

* This paper designs a novel graph message-passing mechanism tailored for multi-task interaction, improving predictive accuracy and model effectiveness.
  
* We integrate a new benchmark dataset from four existing open-source datasets for multi-task traffic prediction.

Recently, multi-task traffic prediction has received increasing attention as it enables knowledge sharing between heterogeneous variables or regions to improve prediction accuracy while meets the prediction requirements from multi-source data in Intelligent Transportation Systems (ITS). However, there are two existing problems in current studies. On one hand, they often tend to construct specialized models for a limited set of predictive parameters, thus lacking in generality. On the other hand, it is challenging to model the graph-based multi-task interaction and message-passing processes due to the heterogeneity of graph structures caused by multi-source traffic data. To address these gaps, this paper proposes a Multi-Graph Interaction Learning Network (MuGIL), which is characterized by three key aspects: 1) A flexible end-to-end multi-task prediction framework, generalizable for varied variables or scenarios; 2) A multi-source graph representation module aligning heterogeneous information through semantic graphs; 3) A novel message-passing mechanism for multi-task graph neural networks, supporting leveraging knowledge among tasks. The model is validated using data from California by comparing with the state-of-the-art prediction models. The results show that the MuGIL model achieves better prediction performance than the state-of-the-art baselines. Ablation experiments demonstrate the importance of the designed multi-source graph representation module and message-passing mechanism.

# Dataest
[Traffic data for multi-task learning](https://pan.baidu.com/s/1GNhLw8NyJJSmTy5ds3FpZA?pwd=ivl3)

extraction code：ivl3

![image](https://github.com/trafficpre/MuGIL/assets/65816926/a9779bf9-cd67-4c13-989c-fd2ae5c0620f)

We construct a benchmark for multi-task traffic prediction, which consists of four real-world datasets collected by the California Deportment of Transportation. From a temporal perspective, all of them contain about two months period traffic flow data and the time window is five minutes. Correspondingly, from a spatial perspective, this data is sourced from different regions within the state of California. Additionally, the geographical information of each region is provided to establish the topological relationship between different traffic observation stations.

# Getting Started

<span id='Code Structure'/>

### 1. Code Structure

```
├── data/
│   ├── pems03.npy
│   ├── pems04.npy
│   ├── pems07.npy
│   └── pems08.npy
├── LibMTL/
│   ├── architecture
│   ├── model/
│   │   └── MuGIL.py
│   ├── weighting
│   ├── config.py
│   ├── loss.py
│   ├── metrics.py
│   ├── trainer.py
│   └── utils.py
├── create_dataset_pems.py
├── README.md
├── requirements.md
├── train.py
├── trainer_pems.py
├── utils.py
└── visual_pems.py
```

<span id='Environment'/>

### 1.Environment
Please first clone the repo and install the required environment.
```shell
conda create -n MuGIL python=3.8.5

conda activate MuGIL

# Torch (other versions are also ok)
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

<span id='Training MuGIL'/>

### 2. Training MuGIL
To train the MuGIL model with all datasets, you can execute the train.py by following code:
```
python train.py
```

### 3. Evaluating
Running Evaluation of MuGIL. The code will read the already trained optimal parameters and evaluate them on the test set. Furthermore, a prediction result visualization function is embedded in the code, allowing you to observe the prediction results by changing the nodeID and the time step you want to plot.
```
python visual_pems.py
```

# Continue to update:
(1) We have made targeted improvements to libMTL, necessitating the generation of a new package for use within this project.

(2) We will provide detailed explanations of key parameters in the code, along with accompanying examples.
