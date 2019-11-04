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
```
net1(  
  (relu): ReLU()  
  (dropout1): Dropout(p=0.2, inplace=False)  
  (fc1): Linear(in_features=784, out_features=512, bias=True)  
  (fc2): Linear(in_features=512, out_features=10, bias=True)  
)    
```
net2: sample tradional net which only has 3 linear fully connected layer.   
```
net2(  
  (relu): ReLU()  
  (dropout1): Dropout(p=0.2, inplace=False)  
  (fc1): Linear(in_features=784, out_features=512, bias=True)  
  (fc2): Linear(in_features=512, out_features=128, bias=True)  
  (fc3): Linear(in_features=128, out_features=10, bias=True)  
)  
```
net3: LeNet which has 2 convlutional layers followed by 2 linear fully connected layer.  
```
LeNet(  
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))  
  (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))  
  (fc1): Linear(in_features=800, out_features=500, bias=True)  
  (fc2): Linear(in_features=500, out_features=10, bias=True)  
  (relu): ReLU()   
) 
```

net4: C3_2F which has 3 convlutional layers followed by 2 linear fully connected layer.  
```
net4(  
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))  
  (conv1_drop): Dropout2d(p=0.5, inplace=False)  
  (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))  
  (conv2_drop): Dropout2d(p=0.5, inplace=False)  
  (conv3): Conv2d(50, 100, kernel_size=(2, 2), stride=(1, 1))  
  (conv3_drop): Dropout2d(p=0.5, inplace=False)  
  (fc1): Linear(in_features=800, out_features=500, bias=True)  
  (fc2): Linear(in_features=500, out_features=10, bias=True)  
  (relu): ReLU()  
)  
```
net5: C3_2F which has 4 convlutional layers followed by 2 linear fully connected layer.  
```
net5(  
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))  
  (conv1_drop): Dropout2d(p=0.5, inplace=False)  
  (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))  
  (conv2_drop): Dropout2d(p=0.5, inplace=False)  
  (conv3): Conv2d(50, 100, kernel_size=(2, 2), stride=(1, 1))  
  (conv3_drop): Dropout2d(p=0.5, inplace=False)  
  (conv4): Conv2d(50, 100, kernel_size=(2, 2), stride=(1, 1))  
  (fc1): Linear(in_features=800, out_features=500, bias=True)  
  (fc2): Linear(in_features=500, out_features=10, bias=True)  
  (relu): ReLU()  
)  
```
net6: C1_B_2F which has 1 convlutional layer, 1 BasicBlock and 2 linear fully connected layer. 
```
net6(  
  (conv1): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2))  
  (conv2): BasicBlock(  
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    (relu): ReLU(inplace=True)  
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  )  
  (fc1): Linear(in_features=760384, out_features=500, bias=True)  
  (fc2): Linear(in_features=500, out_features=10, bias=True)  
)  
```
net7: ResNet18, plaese find more imoformation here about ResNet18 [here](https://arxiv.org/pdf/1512.03385.pdf).    
```
ResNet(  
  (conv1): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (relu): ReLU(inplace=True)  
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)  
  (layer1): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (layer2): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (downsample): Sequential(  
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)  
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      )  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (layer3): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (downsample): Sequential(  
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)  
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      )  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (layer4): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (downsample): Sequential(  
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)  
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      )  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))  
  (fc): Linear(in_features=512, out_features=10, bias=True)  
)  
```
net8: Alexnet, plaese find more imoformation here about Alexnet [here](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).  
```
AlexNet(  
  (features): Sequential(  
    (0): Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))  
    (1): ReLU(inplace=True)  
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)  
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))  
    (4): ReLU(inplace=True)  
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)  
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (7): ReLU(inplace=True)  
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (9): ReLU(inplace=True)  
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (11): ReLU(inplace=True)  
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)  
  )  
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))  
  (classifier): Sequential(  
    (0): Dropout(p=0.5, inplace=False)  
    (1): Linear(in_features=9216, out_features=4096, bias=True)  
    (2): ReLU(inplace=True)  
    (3): Dropout(p=0.5, inplace=False)  
    (4): Linear(in_features=4096, out_features=4096, bias=True)  
    (5): ReLU(inplace=True)  
    (6): Linear(in_features=4096, out_features=10, bias=True)  
  )  
) 
```
net9: MobileNetV2, plaese find more imoformation here about MobileNetV2 [here](https://arxiv.org/pdf/1801.04381.pdf).  
## what I found ##
(1) Different neural networks have different numbers of parameters, then the training time for a single epoch could be >>different.   
(2) For small datasets (such as MNIST), a simple CNN can achieve high classification accuracy. So there may be no need to use a large      neural network. (Using 1000samples to train LeNet 100 epochs takes less time than training ResNet18 25 epochs, but the classification accuracies obtained by both are not much different.)     
(3) For net3 and net4 under the same training conditions, as the networks get deeper, the classification accuracy decreases.    
(4) we introduced the basic block from ResNet into net 6, and the training time for an epoch is much longer compared with net3 and net4.     
(5) For simple CNNs, we need to calculate the size of each feature map then we can specify the structure of the first FC layer. For ResNet, etc., it has a fixed input image size (224 * 224), and a prior resize process is needed.  
  
