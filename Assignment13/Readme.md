In the given folder, 2 jupiter notebooks are submitted.
1. Assignment_13_resnet
2. Assignment_13_custom

The difference in the implementation for both have been explained below

**Assignment_13_resnet**

In this model resnet 18 architecture is followed with small deviations from suggested resnet. Stride used in all the steps is 1 and Maxpooling is not used in any steps. Moreover, 3*3 conv are used in all the steps. 

Please see below the Arch. 

Conv1 --> Block1 --> Block2 --> Block3 --> Block4 --> Global Average Pooling --> Dense Layer --> Softmax
Conv1 = Conv --> Batch Normalization --> Relu
Each Block = One Resnet unit --> Add(Input) --> Relu --> One Resnet unit --> Add(Input) --> Relu
Resnet Unit = Conv --> Batch Normalization --> Relu --> Conv --> Batch Normalization
Conv 1 and Block1 - 64 Channels
Block 2 - 128 Channels
Block 3 - 256 Channels
Block 4 - 512 Channels

**Results**
The target was to cross 92% accuracy but it could not reach 92% accuracy. Though the model crossed 91% validation Accuracy in 20 Epochs.

**Assignment_13_custom**

In this model resnet 18 architecture is followed with more deviations from suggested resnet. Stride used in all the steps is 1 and Maxpooling is not used in any steps. Moreover, 3*3 conv are used in all the steps. 

Please see below the Arch. 

Conv1 --> Block1 --> Block2 --> Block3 --> Block4 --> Block5 --> Block6 --> Block7 --> Block8 -->Global Average Pooling --> Dense Layer --> Softmax
Conv1 = Conv --> Batch Normalization --> Relu --> dropout
Each Block = Conv --> Batch Normalization --> Relu --> dropout --> Conv --> Batch Normalization --> Relu --> dropout --> Add(Input)
Conv 1, Block1 and Block2 - 64 Channels
Block 3, Block4 - 128 Channels
Block 5, Block6 - 256 Channels
Block 7, Block8 - 512 Channels

**Results**
The model crossed 92% validation Accuracy in 41 Epochs. Moreover, this has potential to cross 93% accuracy.

**Suggesions**
Following are the suggestions to improve the custom which I want to try later --
1. Increase dropout from .05 to .1. This will reduce overfitting and can help in improving the accuracy
2. Add cutout in the data augmentation and this can actually help in improving validation accuracy
3. The current learning rate is based on couple of trials to get faster convergance. Using learning rate finder (from assignment 11) can be very helpful in this.
