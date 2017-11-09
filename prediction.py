from skimage import img_as_ubyte
from skimage.io import imread
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, erosion, thin
from keras.models import load_model
import numpy as np
import sys

import cv2

from matplotlib import pyplot as plt

debug = 0

def main(image_abs_path):
    if debug:
        plt.figure(1)

    image = cv2.imread(image_abs_path)
    if debug:
        plt.subplot(231)
        plt.imshow(image)
        plt.title('Original')

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if debug:
        plt.subplot(232)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')

    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 
    if debug:
        plt.subplot(233)
        plt.imshow(thresh, cmap='gray')
        plt.title('Binary image')

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated_img = cv2.dilate(thresh,kernel,iterations = 0)

    if debug:
        plt.subplot(234)
        plt.imshow(dilated_img, cmap='gray')
        plt.title('Dilated')

    binary_inv_img = 255 - dilated_img

    if debug:
        plt.subplot(235)
        plt.imshow(binary_inv_img, cmap='gray')
        plt.title('Binary inverse')

    _,contours, hierarchy = cv2.findContours(binary_inv_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    ans = []
    ret = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if (w>50 or h>50) and cv2.countNonZero(binary_inv_img[y:y+h, x:x+w])/(h*w) < 0.8:
            cv2.rectangle(dilated_img, (x,y), (x+w, y+h), (0,255,0), 10)
            #plt.imshow(thresh[y:y+h, x:x+w], cmap='gray')
            #plt.show()
            ans.append((x,y,w,h))
            #print(x,y,w,h)
    ans.sort()
    for i in ans:
        ret.append(thresh[(i[1]-30):(i[1]+30+i[3]), (i[0]-30):(i[0]+30+i[2])])
    
    if debug:
        plt.subplot(236)    
        plt.imshow(dilated_img, cmap='gray')
        plt.title('Final')
        plt.show()
        
    return ret

"""
for i in range(10):
    main('../../digits/'+str(i)+'.jpg')
"""
def process_img(image):
    resized_img = cv2.resize(image, (100, 100))
    if debug:
        plt.imshow(resized_img, cmap='gray')
        plt.show()

    thinned_img = 1 - thin(255-resized_img)
    
    if debug:
        plt.imshow(thinned_img, cmap='gray')
        plt.show()
    return thinned_img

model = load_model('my_CNN_large_class_weight.h5')
if __name__=='__main__':
    #for i in range(10):
    if len(sys.argv) <= 1:
        print("\n##Error: Please specify the filename.\nCorrect usage: python prediction.py image_path")
        sys.exit()
    op = main(sys.argv[1])
    expr = ""
    for image in op:
        img = process_img(image)
#        plt.imshow(image, cmap='gray')
#        plt.show()
        res = model.predict(np.array([[img]]).astype('float32'))
        ans = 0
        print('+ -> %.2f' % (100*res[0][0]))
        if res[0][ans] < res[0][0]:
            ans = 0
        print('- -> %.2f' % (100*res[0][1]))
        if res[0][ans] < res[0][1]:
            ans = 1
        for i in range(2, len(res[0])):
            print('%d -> %.2f' % (i-2, 100*res[0][i]))
            if res[0][ans] < res[0][i]:
                ans = i
        if ans == 0:
            print('It is: +'+'\n')
            expr += "+ "
        elif ans == 1:
            print('It is: -'+'\n')
            expr += "- "
        else:
            print('The digit is: '+str(ans-2)+'\n')
            expr += str(ans-2)+" "
print("Expression is: "+expr)
print("The result is: ")
print(expr + "= "+str(eval(expr)))
