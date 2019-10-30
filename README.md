# A report for computer vision class #
## Mission ##
Analyse how the structure of a network affects its prediction accuracy and how it depends on the size of training data
## Requirements ##
-Test at least 10 networks that have different structures on MNIST dataset.  
-Train each modelon 1,000 and 50,000 samples until convergence, respectively.    
-Test each model on 10000 test sampls to get mean prdiction accuracy.
## Dependencies ##
> * Python 3.7.3
> * NVIDIA GeForce GTX 1080
> * PyTorch 1.0.1
## Results ##
|Nets            |epochs|1000 samples|epochs|50000 samples|
|:--------------:|:----:|:----------:|:----:|:-----------:|
|net1:2FC_512    |100   |80.48%      |25    |97.71%       | 
|net2:3FC_128    |100   |77.41%      |25    |88.00%       |
|net3:LeNet      |100   |93.01%      |25    |91.53%       |
|net4:C3_2F      |100   |93.12%      |25    |99.19%       |
|net5:C4_2F      |100   |90.49%      |25    |98.80%       |       
|net6:C1_B_2F    |20    |88.00%      |5     |98.19%       |
|net7:ResNet18   |50    |93.62%      |10    |99.24%       |
|net8:Alexnet    |100   |95.31%      |25    |99.49%       |
|net9:MobileNetV2|20    |91.58%      |5     |99.24%       |
## Specific structure for each net ##
net1: sample tradional net which only has 2 linear fully connected layer.  
net2: sample tradional net which only has 3 linear fully connected layer.  
net3: LeNet which has 2 convlutional layers followed by 2 linear fully connected layer.  
net4: C3_2F which has 3 convlutional layers followed by 2 linear fully connected layer.  
net5: C3_2F which has 4 convlutional layers followed by 2 linear fully connected layer.  
net6: C1_B_2F which has 1 convlutional layer, 1 BasicBlock and 2 linear fully connected layer.  
net7: ResNet18, plaese find more imoformation here about ResNet18 [here](https://arxiv.org/pdf/1512.03385.pdf).  
net8: Alexnet, plaese find more imoformation here about ResNet18 [here](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).  
net9:MobileNetV2, plaese find more imoformation here about ResNet18 [here](https://arxiv.org/pdf/1801.04381.pdf).  
