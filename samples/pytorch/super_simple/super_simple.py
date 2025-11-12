import torch
import torch_mlir
import torch.nn as nn
import torch.nn.functional as func

class lenet(nn.Module):
    '''
    input: 3x32x32 image
    output: 10 class probability
    '''
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(1,1,2) #c1:featuremaps 6@28x28 #output = (input-filter)/stride + 1, #filter:5size
        #self.conv2 = nn.Conv2d(6,16,5) #c3:feature_maps 16@10x10
        self.Linear = nn.Linear(3,1) #subsampling 1/2size
        #self.fc1 = nn.Linear(16*5*5,120) #f5:layer120
        #self.fc2 = nn.Linear(120,84) #f6:layer84
        #self.fc3 = nn.Linear(84,10) #output:10 class

    def forward(self,x):
        x = self.Linear(func.relu(self.conv1(x)))
        #x = self.maxPool(func.relu(self.conv2(x)))
        return x

def lenet_main():
    return lenet()

lenet = lenet_main()
lenet.train(False)


# Compile the model with torch_mlir
# Note: LeNet expects RGB input (3 channels), so using 3x32x32 for CIFAR-like input
module = torch_mlir.compile(lenet, torch.ones(1, 1, 4, 4), output_type="linalg-on-tensors")
print(module)

# Alternative: for larger input size (32x32 as in original LeNet paper)
# module = torch_mlir.compile(LeNet(), torch.ones(1, 1, 32, 32), output_type="linalg-on-tensors")
# print(module)

# traced_script_module = torch.jit.trace(model, torch.ones(1, 1, 28, 28))
# traced_script_module.save("lenet_model.pt")