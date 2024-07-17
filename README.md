# Evaluation of Classification algorithms for Distributed Denial of Service Attack Detection

## Folders

- The repository is organised as follows:  
    - `CICDDoS2019/`: contains the dataset and preprocessing results.
    - `LoadData/`: contains scripts for preprocessing and encoding.
    - `Baseline/`: contains five baseline model and related experiments configures.

## Dependencies

- python 3.8
- numpy 1.22.4
- pandas 2.0.3
- torch 1.11.0+cu113
- scikit-learn 1.3.2

## Data

- I used the open-source dataset `CICDDoS2019` mentioned in the paper, and I have selected top 25 features for training the models. The features are as follows:
>   Unnamed: 0    
    Flow ID    
    Source IP    
    Source Port   
    Destination IP    
    Destination Port    
    Timestamp       
    Fwd Packet Length Min 
    Bwd Packet Length Max    
    Bwd Packet Length Min    
    Bwd Packet Length Mean  
    Flow Bytes/s   
    Flow Packets/s    
    Fwd PSH Flags    
    Fwd Packets/s    
    Min Packet Length    
    Packet Length Std    
    RST Flag Count   
    ACK Flag Count  
    URG Flag Count    
    CWE Flag Count   
    Down/Up Ratio   
    Avg Fwd Segment Size   
    Avg Bwd Segment Size  
    Inbound

- After preprocessing, the following files will be generated:   
    - `Data.npz`: contains the all dataset. 
    - `train.npz`: contains the training features and labels.
    - `test.npz`: contains the testing features and labels.

## Models

- `Decision Tree`: builds a tree structure where the leaf nodes represent the classification result.
- `Naive Bayes`: assumes independence between features based on the Bayes' theorem, calculates the probability of each class and selects the class with the highest probability as the prediction result.
- `Logistic Regression`: maps input features to a logistic function which outputs a probability between 0 and 1, and uses this probability to make the prediction result.
- `K Nearest Neighbor`: calculates the distance between the samples in the testing dataset and each sample in the training set, selects the K nearest neighbors and predicts the classification result through voting.
- `Random Forest`: constructs multiple decision tree models and aggregates their predictions through voting.

## Run

### DecisionTree

- train DecisionTree model by running:    
    `python DecisionTree.py --MaxDepth 20 --MinSample 25`

### KNN

- train KNN model by running:      
    `python KNN.py --K 100`

### LogisticRegression

- train LogisticRegression model by running:     
    `python LogisticRegression.py --batch_size 1000 --epochs 20 --feature_dis 25 --lr 0.001`

### NBClassifier

- train NBClassifier model by running:    
    `python NBClassifier.py`

### RandomForest

- train RandomForest model by running:    
    `python RandomForest.py --num 100 --ratio 1 --k 5 --MaxDepth 20 --MinSample 25`

## Result

|Model|Accuracy|Precision|Recall|F1-score|
|:---:|:---:|:---:|:---:|:---:|
|Decision Tree|0.995|0.991|1|0.995|
|Naive Bayes|0.995|0.991|0.998|0.995|
|Logistic Regression|0.879|0.856|0.913|0.883|
|K Nearest Neighbor|0.999|0.999|0.999|0.999|
|Random Forest|0.998|0.995|0.999|0.998|

## Feedback

If you need help or find any bugs, feel free to submit GitHub issues or PRs.
