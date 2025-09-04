Explain
==================
1.) The project is working on creating multi-agent reinforcement learning algorithms.
2.) The goal is to deploy these reinforcement learning algorithms onto NVIDIA Jetson Nano Hardware and emulate racing cars.
3.) 


How To Use training and testing scripts
===================
  Assuming that donkeycar has been setup with installation from the open source documentation and githubs. 
1.) In ubuntu terminal, run nano train.py(or a similar command) to create an empty space and copy and paste in the algorithm.
2.) Then, run python and the name of your training script to start a training series: python train.py
3.) Use the training script and modify its value to include your best testing model. The testing model included has 220,000 steps because that was my best model.

Additional Notes
==================
1.) More is intended to be added as progress is made.
2.) One set of a training and testing script is for the waveshare map, the other is for the warehouse map.
3.) We believe it is not possible to deploy onto the jetson nano using the waveshare map which is why we want to move to testing with the warehouse map.

Current progress
=================
1.) Developed algorithm for single car for both waveshare and warehouse map
2.) Currently working on improving it and then further adding in another to emulate chasing.
3.) Hoping to deploy on Jetson Nano soon.
