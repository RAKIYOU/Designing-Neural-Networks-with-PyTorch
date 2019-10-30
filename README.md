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
|Nets            |epochs|1000 samples|epochs|50000 samples|
|:--------------:|:----:|:----------:|:----:|:-----------:|
|Net1:2FC_512    |100   |80.48%      |25    |97.71        | 
|Net2:3FC_128    |100   |77.41%      |25    |88.00%       |
|Net3:LeNet      |100   |93.01%      |25    |91.53%       |
|Net4:C3_2F      |100   |93.12%      |25    |99.19%       |
|Net5:CNN2       |100   |90.49%      |25    |98.80%       |       
|Net6:CNN3       |20    |88.00%      |5     |98.19%       |
|Net7:ResNet18   |50    |93.62%      |10    |99.24%       |
|Net8:Alexnet    |100   |95.31%      |25    |99.49%       |
|Net9:MobleNet   |20    |91.58%      |5     |99.24%       |
