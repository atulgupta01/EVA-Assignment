Please find below the explanation

init_pytorch - the function is used for initializing the parameter values. Below is given the logic for initialization
1. Calculate the product of all the values except the last element - the value is called fan
2. Take square root of the value calculated in step one - the value is called bound
3. Initialize the same number of values between negative bound and positive bound randomly

class ConvBN - The class is used to initialize the layes for the DNN
1. Add a convolution layer with kernels initialized using init_pytorch
2. Add Batch Normalization layer (with momentum and epsilon)
3. Add dropout layer with 5% dropout
4. Sequence layers in following order
	a. Convolution Layer
	b. dropout
	c. Batch Normalization
	d. Relu
	
