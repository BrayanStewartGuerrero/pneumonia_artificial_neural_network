from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# Definir hiperparámetros
altura, longitud = 224, 224
batch_size = 32
filtros_Conv1 = 32
filtro1_size = (3,3)
filtros_Conv2 = 64
filtro2_size = (3,3)
filtros_Conv3 = 128
pool_size = (2,2)
clases = 3
lr = 0.0001
epocas = 50

# Directorios de datos
data_train = '/path/to/train'
data_val = '/path/to/val'
data_test = '/path/to/test'

# Generador de imágenes para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_train,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Generador de imágenes para el conjunto de validación
val_datagen = ImageDataGenerator(
    rescale=1./255
)

val_generator = val_datagen.flow_from_directory(
    data_val,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Cargar modelo VGG16 pre-entrenado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(altura, longitud, 3))

# Congelar capas de VGG16
for layer in base_model.layers:
    layer.trainable = False

# Añadir capas personalizadas al modelo
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
output = Dense(clases, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compilar modelo
optimizer = Adam(lr=lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Definir callbacks
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
tb = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

# Entrenar modelo
history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epocas, 
                              validation_data=val_generator, validation_steps=len(val_generator),
                              callbacks=[es, mc, tb])

# Graficar entrenamiento
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.show()

# Graficar la curva de pérdida
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Curva de pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.show()

# Graficar la curva de precisión
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Curva de precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.show()
