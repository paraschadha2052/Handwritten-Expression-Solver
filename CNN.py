import numpy
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

def make_less(data_set, d, size=900):
    dct = {}
    lst = []
    for i in range(12):
        dct[i] = 0
    for tst in data_set:
        for i in range(10):
            if tst['label'][i]:
                break
        if i==d:
            if dct[i]<size:
                lst.append(tst)
                dct[i] += 1
        else:
            lst.append(tst)
    return lst

with open('train/train.pickle', 'rb') as train:
    print('Restoring training set ...')
    train_set = pickle.load(train)
train_set = make_less(train_set, 7, 500)

with open('test/test.pickle', 'rb') as test:
    print('Restoring test set ...')
    test_set = pickle.load(test)
    
num_pixels = len(train_set[0]['features'])
num_classes = len(train_set[0]['label'])

X_train = numpy.zeros(shape=(len(train_set), len(train_set[0]['features']))).astype('float32')
y_train = numpy.zeros(shape=(len(train_set), len(train_set[0]['label'])))
for i in range(len(train_set)):
    X_train[i] = train_set[i]['features']
    y_train[i] = train_set[i]['label']
    
X_test = numpy.zeros(shape=(len(test_set), len(test_set[0]['features']))).astype('float32')
y_test = numpy.zeros(shape=(len(test_set), len(test_set[0]['label'])))
for i in range(len(test_set)):
    X_test[i] = test_set[i]['features']
    y_test[i] = test_set[i]['label']

X_train = X_train.reshape(X_train.shape[0], 1, 100, 100).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 100, 100).astype('float32')

def larger_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 100, 100), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = larger_model()
class_weight = numpy.array([ 0.50577648,  0.36789031,  1.86921348,  0.48003232,  0.48360465,
        1.1822058 ,  1.77054066,  3.63867017,  4.2010101 ,  4.49621622,
        4.30093071,  4.16316316])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=14, shuffle=True, batch_size=100, verbose=2, class_weight=class_weight)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save('my_CNN_large_class_weight_new.h5')
