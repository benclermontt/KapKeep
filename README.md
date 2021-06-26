# KapKeep
KapKeep is a web-based service which will monitor the occupancy of a building to ensure that they are following their occupancy requirements. This will be accomplished by having cameras positioned at the entrance(s) and exit(s) of the building. These cameras will use our own implementation of a Histogram of Oriented Gradients body detection algorithm using Linear SVM for the vector comparison. 

Package Dependence
OpenCV: pip3 install opencv-python
PiCamera: pip3 install picamera[array]

Our current socket server will pass a maximum of 5 frames per second to the server
