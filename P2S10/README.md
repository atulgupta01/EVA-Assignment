Twin Delayed Deep Deterministic Gradient

Github Link for the code 

https://github.com/atulgupta01/EVA-Assignment/tree/master/P2S10

You Tube Link

https://youtu.be/MTY2TE-1qoU

The assignment has following files

ai.py - It has replay buffer class, actor class, critic class and TD3 class
map.py - It has environment related details with CAR class, CITY class and environment class
P2S10.ipynb - This notebook is run on colab to train the model
inference.py - This file is used for testing/inference of the model.

Challenges Faced during the process

1. It required proper thought process to setup the environment. However, I did setup only the bare minimum skelton of the environment. Some of the functionality in AI Gym environment are not made available i.e. providing action space etc.

2. The training process of the program was very long and required patience to re-run everytime after making the changes

3. Some variable were not considered important initially but after changing find of great importance i.e. _max_episode_steps = 500.
This helped in removing the overfitting in the environment

4. I could setup and run the environment in the first week but could not train to the level I wanted to. It has lot of moving parts and changing multiple parts at the same time, deteriorates the things to worst level

5. Faced mutiple problems and similar problems were faced by people in the group. Some of the problems were more of carelessness i.e. in step 5 I passed current state instead of next state. It took 3-4 days to figure out the issue and resolve it.

Current Status

1. Trained with the fixed 3 points and was able to get almost 90% level of success
2. Trained with 3 randomly moving points and was able to get 70% some level of success
3. Trained with 7 randomly moving points and success was not very good. The model was not able to learn very well.

What can be done to improve

1. Change the reward system to improve the on road percentage of the car.
2. Add random rotation to state images - i feel my model is over fitting and this can reduce overfitting
3. Currently variable _max_episode_steps is setup as 500. Try reducing the value to 400 avoid overfitting.
4. Try rotating the state image similar to Gaurav Patel's model and see if that helps

Expectation from Evaluation

1. It will be a huge help if you can please point my mistakes during evaluation. I would like to continue working on this even after submitting the assignment. So, any guidance will be really helpful.

2. At times, I felt running out of ideas to improve and any suggestion will be really welcome.

3. Could we also get some more similar ideas where we can create environment and play for our learning. I shall be thankful if you can please point me to more such ideas.
