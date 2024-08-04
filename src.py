from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import time
import picamera
from picamera.array import PiRGBArray
from sense_hat import SenseHat

# Loading Trained Model
model = keras.models.load_model('convnet_model.h5')

# Init PiCamera
camera = picamera.PiCamera()
rawCapture = PiRGBArray(camera)
time.sleep(0.2)

# Init Sense Hat
s = SenseHat() 
s.set_rotation(90)

def ImagePreProcess(file_path):
    global acc # set global else locality error
    acc = 0
    # Read in Gray Scale Image
    orig_img = cv2.imread(file_path, 0) 
    # Invert Image Values
    image_gray_invert = 255 - orig_img
    # blur = cv2.GaussianBlur(image_gray_invert, (5,5), 0)

    # Applying Otsu's Thresholding
    (ret, image_bw) = cv2.threshold(image_gray_invert, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    img_resized = cv2.resize(image_bw, (28,28), interpolation=cv2.INTER_CUBIC)

    # Reshaping Image for Model Input
    test_img = img_resized.reshape(1, 28, 28, 1)
    test_img = test_img.astype('float32')
    test_img /= 255

    cv2.imwrite('/home/pi/Desktop/gray_invert.jpg', image_gray_invert)
    cv2.imwrite('/home/pi/Desktop/image_blackwhite.jpg', image_bw)

    # Make Prediction
    a = model.predict(test_img)
    ans = a[0].tolist().index(max(a[0].tolist()))
    a = a[0].tolist()
    for i, val in enumerate(a):
        if val > .5:
            acc = a[i]
            a[i] = 1
        else:
            a[i] = 0

    print(a)
    print('Predicted digit is:', ans)
    print('Probability of', acc*100)

    # Display Predicted Digit on Sense Hat if accuracy if above 50%
    if acc > .5:
        s.show_letter(str(ans))
    else:
        s.clear()


def main():
    try:
        while True:
            # Capture RGB image and set it numpy array
            camera.capture(rawCapture, format='bgr')
            orig_img = rawCapture.array
            cv2.imwrite('/home/pi/Desktop/img.jpg', orig_img)

            # Resize Window Frame and Display Image
            cv2.namedWindow('Frame', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Frame', orig_img)
            orig_img = cv2.resizeWindow('Frame', 300, 300)

            file_path = '/home/pi/Desktop/img.jpg'
            ImagePreProcess(file_path)

            # allows for the capture of multiple array by empting array each capture
            rawCapture.truncate(0)

            cv2.waitKey(delay=1) # wait 1ms for next frame

    except KeyboardInterrupt:
        s.clear()
        cv2.destroyAllWindows()
        camera.close()

if __name__ == "__main__":
    main()
