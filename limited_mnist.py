#!/usr/bin/env python
# coding: utf-8

# In[65]:


#####################################################
### --- We modify the dataloader in this block ---###
#####################################################

from __future__ import print_function
import torchvision.datasets.vision as vision
import warnings
from PIL import Image
import os
import os.path
import random
import numpy as np
import torch
import codecs
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive,     makedir_exist_ok, verify_str_arg

class MyMNIST(vision.VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, limit_data=None, resize=False):
        super(MyMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        # ------- We can ingore this block -------
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        # -----------------------------------------------

        # We change the lines below; these specify how the data are loaded.
        # self.data contain images (type 'torch.Tensor' of [num_images, H, W]) 
        # self.targets contain labels (class ids) (type 'torch.Tensor' of [num_images]) 
        # images and labels are stored in the same 
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        # We use only the images and lables whose indeces are in the range of 0..limit_data-1.
        if not limit_data is None:
          self.data    = self.data[ :limit_data, :,:]
          self.targets = self.targets[:limit_data]
          if self.train:
            print("[WRN]: Trainig Data is limited, only the first "+str(self.data.size(0))+" samples will be used.")
          else:
            print("[WRN]: Test Data is limited, only the first "   +str(self.data.size(0))+" samples will be used.")            


    def __getitem__(self, index):
        # We extract the image and label of the specified 'index'.
        img, target = self.data[index], int(self.targets[index])

        # Prepare for self.transform below.
        img = Image.fromarray(img.numpy(), mode='L')

        # Transform img.
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))
        
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


# In[66]:


#####################################################
### --- Define a simple network in this block --- ###
#####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

# Build a simple network
class simple_network(nn.Module):
  def __init__(self, input_dim=1, num_class=10):
    super(simple_network, self).__init__()

    # Fully connected layers
    self.fc1 = nn.Linear(784, 512)
    #nn.init.kaiming_uniform_(self.fc1.weight)
    self.fc2 = nn.Linear(512, num_class)
    #nn.init.kaiming_uniform_(self.fc2.weight)
    
    # Activation func.
    #self.relu = nn.ReLU()

    #self.dropout1 = nn.Dropout(0.2)
    
  def forward(self, x):
    #x = self.relu(self.fc1(x))                   print("Trainig Data is limited, only the first "+str(self.data.size(0))+" samples are used.")
    #x = self.relu(self.fc2(x))     
    x = F.relu(self.fc1(x))  
    #x = self.dropout1(x)
   # x = F.relu(self.fc2(x))     

    return x
  


# In[67]:


############################################
### --- Define a LeNet in this block --- ###
############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define LeNet
class LeNet(nn.Module):
  def __init__(self, input_dim=1, num_class=10):
    super(LeNet, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(input_dim, 20,  kernel_size=5, stride=1, padding=0)
    self.conv2 = nn.Conv2d(    20,    50,  kernel_size=5, stride=1, padding=0)

    # Fully connected layers
    self.fc1 = nn.Linear(800, 500)
    self.fc2 = nn.Linear(500, num_class)
    
    # Activation func.
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))                         # Conv.-> ReLU
    x = F.max_pool2d(x, kernel_size=2, stride=2)         # Pooling with 2x2 window
    x = self.relu(self.conv2(x))                         # Conv.-> ReLU
    x = F.max_pool2d(x, kernel_size=2, stride=2)         # Pooling with 2x2 window

    b,c,h,w = x.size()                                   # batch, channels, height, width
    x = x.view(b, -1)                                    # flatten the tensor x

    x = self.relu(self.fc1(x))                           # fc-> ReLU
    x = self.fc2(x)                                      # fc
    return x


# In[68]:


############################################
### --- Define CNN1 in this block --- ###
############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define CNN1
class CNN1(nn.Module):
  def __init__(self, input_dim=1, num_class=10):
    super(CNN1, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(input_dim, 20,  kernel_size=5, stride=1, padding=0)  
    self.conv1_drop = nn.Dropout2d()
    self.conv2 = nn.Conv2d(    20,    50,  kernel_size=5, stride=1, padding=0)   
    self.conv2_drop = nn.Dropout2d()
    self.conv3 = nn.Conv2d(    50,    100,  kernel_size=2, stride=1, padding=0) 
    self.conv3_drop = nn.Dropout2d()
    
    # Fully connected layers
    self.fc1 = nn.Linear(800, 500)
    self.fc2 = nn.Linear(500, num_class)
    
    # Activation func.
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))                                   # Conv.-> ReLU
    #print(x.size())
    x = F.max_pool2d(x, kernel_size=2, stride=2)                  # Pooling with 2x2 window
    #print(x.size())
    x = self.relu(self.conv2(x))                                  # Conv.-> ReLU
    #print(x.size())
    x = F.max_pool2d(x, kernel_size=2, stride=2)                  # Pooling with 2x2 window
    #print(x.size())

    b,c,h,w = x.size()                                            # batch, channels, height, width
    #print(x.size())
    x = x.view(b, -1)                                             # flatten the tensor x
    #print(x.size())

    x = self.relu(self.fc1(x))                                    # fc-> ReLU
    x = self.fc2(x)                                               # fc
    return x


# In[69]:


############################################
### --- Define ResNet in this block --- ###
############################################

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self,in_channel,num_channel,use_conv1x1=False,strides=1):
        super(Residual,self).__init__()
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm2d(in_channel,eps=1e-3)
        self.conv1=nn.Conv2d(in_channels =in_channel,out_channels=num_channel,kernel_size=3,padding=1,stride=strides)
        self.bn2=nn.BatchNorm2d(num_channel,eps=1e-3)
        self.conv2=nn.Conv2d(in_channels=num_channel,out_channels=num_channel,kernel_size=3,padding=1)
        if use_conv1x1:
            self.conv3=nn.Conv2d(in_channels=in_channel,out_channels=num_channel,kernel_size=1,stride=strides)
        else:
            self.conv3=None


    def forward(self, x):
        y=self.conv1(self.relu(self.bn1(x)))
        y=self.conv2(self.relu(self.bn2(y)))
        # print (y.shape)
        if self.conv3:
            x=self.conv3(x)
        # print (x.shape)
        z=y+x
        return z

# blk = Residual(3,3,True)
# X = Variable(torch.zeros(4, 3, 96, 96))
# out=blk(X)

def ResNet_block(in_channels,num_channels,num_residuals,first_block=False):
    layers=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            layers+=[Residual(in_channels,num_channels,use_conv1x1=True,strides=2)]
        elif i>0 and not first_block:
            layers+=[Residual(num_channels,num_channels)]
        else:
            layers += [Residual(in_channels, num_channels)]
    blk=nn.Sequential(*layers)
    return blk


class ResNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(ResNet,self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2,padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.block2=nn.Sequential(ResNet_block(64,64,2,True),
                                  ResNet_block(64,128,2),
                                  ResNet_block(128,256,2),
                                  ResNet_block(256,512,2))
        self.block3=nn.Sequential(nn.AvgPool2d(kernel_size=3))
        self.Dense=nn.Linear(512,10)


    def forward(self,x):
        y=self.block1(x)
        y=self.block2(y)
        y=self.block3(y)
        b,c,h,w = y.size()                                   # batch, channels, height, width
        #x = x.view(b, -1)    
        
        y=y.view(b,-1)
        y=self.Dense(y)
        return y


'''
    
    
    


import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    


# In[71]:


##########################################################################
### Prepare the trainloader/testloader for training/validating network ###
##########################################################################

from   torchvision import datasets as datasets
import torchvision.transforms as transforms
import torch.utils as utils
import matplotlib.pyplot as plt
import torch
import torchvision

# The 「transform」 is used to 
# i)  convert PIL.Image to torch.FloatTensor (batch, dim, H, W), and change the 
#     inputs' range to [0, 1] (by inputs/= 255.0);
# ii) standardize the input images by mean=0.1307, std=0.3081
#transform = transforms.Compose([  transforms.Resize((224,224)),  transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# Initialize dataloaders.
#limit_data  = 1000  # The first $limit_data samples
mnist_train = MyMNIST('./data', train=True,  download=True, transform=transform)
#mnist_train = MyMNIST('./data', train=True,  download=True, transform=transform, limit_data=1000)
mnist_test  = MyMNIST('./data', train=False, download=True, transform=transform)
trainloader = utils.data.DataLoader(mnist_train, batch_size=50, shuffle=True,  num_workers=4)
testloader  = utils.data.DataLoader(mnist_test, batch_size=50, shuffle=False, num_workers=4)


# In[72]:


# Initialize the network
net = simple_network().cuda()
#net = LeNet().cuda()
#net = CNN1().cuda()
#net=ResNet(1,10).cuda()
net_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}
model = ResNet(**net_args)
#print(net)


# In[73]:


############################################
### Prepare the optimizer and loss func. ###
############################################

import torch.optim as optim
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters())
epoch = 0


# In[74]:


######################################################
#### ------ Script for evaluate the network ------ ###
######################################################

def evaluate_model():
  print("Testing the network...")
  net.eval()
  total_num   = 0
  correct_num = 0
  for test_iter, test_data in enumerate(testloader):
    # Get one batch of test samples
    inputs, labels = test_data    
    bch = inputs.size(0)
    print(bch)
    #inputs = inputs.view(bch, 1,28,28)
   
    # Move inputs and labels into GPU
    inputs = inputs.cuda()
    labels = torch.LongTensor(list(labels)).cuda()

    # Forward
    # int(inputs)
    outputs = net(inputs)   

    # Get predicted classes
    _, pred_cls = torch.max(outputs, 1)
#     if total_num == 0:
#        print("True label:\n", labels)
#        print("Prediction:\n", pred_cls)
#     # Record test result
    correct_num+= (pred_cls == labels).float().sum().item()
    total_num+= bch
  net.train()
  
  print("Accuracy: "+"%.5f"%(correct_num/float(total_num)))


# In[75]:


#####################################################
### ------ Script for training the network ------ ###
#####################################################

epoch_size = 20
for epoch_idx in range(epoch_size):
  running_loss = 0.0
  ct_num       = 0
  for iteration, data in enumerate(trainloader):
    # Take the inputs and the labels for 1 batch.
    inputs, labels = data
    #print(length(inputs))
    bch = inputs.size(0)

    inputs = inputs.view(bch, -1)
    #inputs = inputs.unsqueeze(-1)
    

    # Move inputs and labels into GPU
    inputs = inputs.cuda()
    labels = labels.cuda()

    # Remove old gradients for the optimizer.
    optimizer.zero_grad()

    # Compute result (Forward)
    print(inputs.size())
    outputs = net(inputs)
    
    # Compute loss
    loss    = loss_func(outputs, labels)

    # Calculate gradients (Backward)
    loss.backward()

    # Update parameters
    optimizer.step()
    
    #with torch.no_grad():
    running_loss += loss.item()
    ct_num+= 1
    if iteration%20 == 19:
      print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')

    # Test
    if epoch%3 == 2 and iteration%100 == 99:
      evaluate_model()
    
  epoch += 1



evaluate_model()




