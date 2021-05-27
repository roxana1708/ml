from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, GlobalMaxPooling2D, MaxPooling2D, Dropout, Convolution2D, Conv2DTranspose
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
import itertools

x_train = []
y_train = []
x_test = []
y_test = []

# formam vectorii pentru antrenament
with open("train.txt") as filename:
    for current_line in filename:
        line = current_line.split(",")
        img = image.imread("train/" + line[0])
        imgg = asarray(img)
        x_train.append(imgg)
        y_train.append(int(line[1]))

# formam vectorii de testare
with open("validation.txt") as filename:
    for current_line in filename:
        line = current_line.split(",")
        x_test.append(asarray(image.imread("validation/" + line[0])))
        y_test.append(int(line[1]))

# convertim vectorii in vectori numpy
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

#
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


#
input_train = x_train.reshape(len(x_train), x_train[0].shape[0], x_train[0].shape[1], 1)
input_test = x_test.reshape(len(x_test), x_test[0].shape[0], x_test[0].shape[1], 1)

# am ales sa construiesc modelul cu sequential
model = Sequential()

model.add(Conv2DTranspose(50, (5, 5), activation='relu', input_shape=(50, 50, 1))) #elu #gelu
model.add(MaxPooling2D(pool_size=(2, 2))) #pool_size=(2, 2)

model.add(Conv2DTranspose(50, (5, 5), activation='relu')) #gelu
model.add(MaxPooling2D(pool_size=(2, 2))) #pool_size=(2, 2)

model.add(Conv2D(50, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #pool_size=(2, 2)



model.add(Flatten())


model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(750, activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(500, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dense(3, activation='softplus')) # softplus

# pregatim pentru antrenament
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# antrenam modelul
hist = model.fit(input_train, y_train_one_hot,
                 batch_size=128,
                 epochs=13,
                 validation_split=0.25)


# obtinem acuratetea
print(model.evaluate(input_test, y_test_one_hot)[1])


# salvam predictiile
x_test = []
with open("test.txt") as filename:
    for current_line in filename:
        x_test.append(asarray(image.imread("test/" + current_line.rstrip())))

x_test = np.array(x_test)
input_test = x_test.reshape(len(x_test), x_test[0].shape[0], x_test[0].shape[1], 1)

predictions = model.predict(input_test)
classes = np.argmax(predictions, axis=1)
print(classes[0])
print(classes[1])
print(classes[2])
print(classes[len(classes)-3])
print(classes[len(classes)-2])
print(classes[len(classes)-1])


# scriem predictiile modelului in fisier
output = open("predictions.csv", "w")
count = 0
writer = csv.writer(output, delimiter=',')
with open("test.txt") as filename:
    for current_line in filename:
        writer.writerow([current_line.rstrip(), classes[count]])
        count += 1

output.close()
print(count)
print(len(classes))


# salvam modelul
model.save('models/final_model', save_format='h5')

# facem matricea de confuzie
y_predictions = model.predict(input_test)
y_predictions = np.argmax(y_predictions, axis=1)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_predictions)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matricea de Confuzie')
plt.ylabel('Clasa reala')
plt.xlabel('Clasa prezisa')
plt.colorbar()
plt.tight_layout()

plt.xticks(np.arange(3), [0, 1, 2], rotation=45)
plt.yticks(np.arange(3), [0, 1, 2])

thresh = conf_matrix.max()/2
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, conf_matrix[i][j], horizontalalignment="center", color="white" if conf_matrix[i][j] > thresh else "black")

plt.show()
