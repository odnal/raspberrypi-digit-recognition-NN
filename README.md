This project implementats a real-time handwritten digit recognition system on the Raspberry Pi 
using the MNIST dataset. The implementation uses a convolutional neural network to learn the process 
of recognizing digits in this multi-binary classification system. Once the model is trained, using 
the Raspberry Pi, the Pi Camera module, and my own set of handwritten digits the model is tested in 
real-time on the prediction of the images the Pi Camera is capturing. The output, if predicted correctly 
will correspond to the image that is being captured. This project also used the Sense Hat which is another 
compatible component for the Raspberry Pi that consists of a convenient led matrix. The Sense Hat is used 
as a nice visual to display the digit on the board if the model predicted the digit correctly.


More on the MNIST dataset can be found here - [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)

### Digit Capture with Accuracy 
![Digit Capture with Accuracy](https://github.com/landonbr/raspberrypi-digit-recognition-NN/blob/master/pics/Picture1.png)


### Sense Hat Output
![Sense Hat Output](https://github.com/landonbr/raspberrypi-digit-recognition-NN/blob/master/pics/Picture2.jpg)


These are some important modules that are neccessary to the CNN model. And a few others that
are specific to this project alone.

> pip install numpy
> pip install tensorflow
> pip install opencv-python
> pip install picamera
> pip install sense-hat

Don't forget to create and activate a virtual environment before downloading the packages:

> python3 -m venv "path"
> source "path"/venv/activate
