import tensorflow as tf
from tensorflow import keras

# MNIST, "Modified National Institute of Standards and Technology" (Değiştirilmiş Ulusal Standartlar ve Teknoloji Enstitüsü) tarafından oluşturulan 
# bir veri kümesidir. MNIST veri kümesi, el yazısı rakamlarının görüntülerini ve ilgili etiketleri içerir.
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
