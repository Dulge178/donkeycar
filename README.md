Explain
==================
This project is focused on multi-agent reinforcement learning through the Donkey Car simulator and then deploying the created models onto an NVIDIA Jetson Nano to create a testbed for AI Smartcar Racing. There are training and testing scripts listed here that can be used for a variety of purposes.


How To Use training and testing scripts to create the native/mimic two car racing model
===================
You need to setup your config.py to have the waveshare-v0 map because that was the map we used for our models for the two car racing simulation. The training script titled 'train_onewave.py' is for training the faster car and the model for deploying it is the script 'test_onewave.py'. The training script titled 'train_twowave.py' is the training for the slower car and the model for deploying it is the script 'test_twowave.py'. The two testing scripts should be run together and it will put both cars in the same simulation and they can race alongside each other. Currently, they constantly collide due to the size of the track, which is why we are still working on it. 

Additional Notes
==================


Current progress
=================
We have developed a native/mimic two car racing simulation by using two respectively trained models; one that is for a fast car and one that is for a slow car. Then, we deploy them and run them such that they are on the same port and run in the same simulation. Sadly, the faster car crashes into the slower car, but it is able to steer around it and overtake in some cases. Our next steps are to create real multi-agent reinforcement learning and generating more advanced models to do so. 
