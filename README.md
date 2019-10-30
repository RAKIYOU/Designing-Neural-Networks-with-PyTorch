# A report for computer vision #
## Mission ##
Analyse how the structure of a network affects its prediction accuracy and how it depends on the size of training data
## requirements ##
-Test at least 10 networks (models) that have different structures.
-Train each modelon 1,000 and 50,000 samples until convergence, respectively.    
-Test each model on 10000 test sampls to get mean prdiction accuracy.
## Dependencies ##
> * Python 3.7.3
> * NVIDIA GeForce GTX 1080
> * PyTorch 1.0.1
## Results ##
|CNNs        |epochs|1000 samples|epochs|50000 samples|
|:----------:|:----:|:----------:|:----:|:-----------:|
|2FC_512     |100   |80.48%      |25    |97.71        | 
|3FC_128     |100   |77.41%      |25    |88.00%       |
|LeNet       |100   |93.01%      |25    |91.53%       |
|C3_2F       |100   |93.12%      |25    |99.19%       |
|CNN2        |100   |90.49%      |25    |98.80%       |       
|CNN3        |20    |88.00%      |5     |98.19%       |
|ResNet18    |50    |93.62%      |10    |99.24%       |
|Alexnet     |100   |95.31%      |25    |99.49%       |
|MobleNet    |20    |0.9158%     |5     |99.24%       |
