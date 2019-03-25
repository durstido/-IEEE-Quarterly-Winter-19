from keras.models import load_model
import numpy as np
import cv2

#load the model
model = load_model('/Users/durstido/PycharmProjects/IEEEQP_WI19/model.h5')

#NN will process an image, convert it to 150x150,
#and process it through the model to get a classification
def NN(image):
    img = cv2.imread(image, 0)
    try:
        img.shape
        cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (150, 150), interpolation=cv2.INTER_CUBIC)
        #predict 
        pred = model.predict(img)
        pred = np.argmax(pred,axis=1)[0]
        if pred == 0:
            return 0
        elif pred == 1:
            return 1
        elif pred == 2:
            return 2
        elif pred == 3:
            return 3
    except AttributeError:
        print(image + " shape not found")