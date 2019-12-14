tf.enable_eager_execution() - using eager execution code can be executed without a session. This helps in easier debugging of the session.

init_pytorch - the function is used for initializing the parameter values. Below is given the logic for initialization
Input Parameters - 
1. Shape = size of the input tensor i.e. [2,2] represents there are 2 rows and 2 columns
2. Data Type = default value is float but this is the data type for the the random initialization

1. Calculate the product of all the values except the last element of the shape variable - the value is called fan. i.e. if the shape is [3,3], fan will be 3 
2. Take square root of the value calculated in step one and divide one by this new value - the value is called bound. i.e. as per step 1, bound will be 1/(3*3)
3. Initialize the array with same shape as per the shape variable and the values of the array will be between negative bound and positive bound randomly. This array is returned as parameter from the function 

class ConvBN - The class is used to initialize the layes for the DNN
1. Add a convolution layer with kernels initialized using init_pytorch
2. Add Batch Normalization layer (with momentum and epsilon)
3. Add dropout layer with 5% dropout
4. Sequence layers in following order
	a) Convolution Layer
	b) dropout
	c) Batch Normalization
	d) Relu
	
class ResBlk - The class creates a block for the resnet using a parameter res
1. In case res = False, it adds Maxpool layer to ConvBN set of layers and returns it.
2. In case res = True, it adds additional two ConvBN set of layers to last layer (which has maxpool on ConvBN set of layers). In this case before returning it, adds last layer output to new layer output, making it a skip connection.

class DavidNet - It add multiple ConvBN Layers and ResBlk Blocks in following order
1. Add a ConvBN with the defined kernels e.g. number of kernels (c)  = 64 for assuming
2. Add one ResBlk with res=True and kernels = c*2 i.e. 128
3. Add one ResBlk with res=False and kernels = c*4 i.e. 256
4. Add one ResBlk with res=True and kernels = c*8 i.e. 512
5. Add GlobalMaxPool layer
6. Add dense layer to it
7. Add softmax for prediction
8. Add loss function

- In the next step - download Cifar10 dataset
- 
