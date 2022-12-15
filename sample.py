import pyaudio as pyaudio
import tensorflow as tf

import os
from os.path import exists

import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
b_training = True
n_steps = 0

image = [ ]
list_image = [ ]
list_label = [ ]
list_label_note = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G' ]

for i in range(80):
	list_image.append(tf.zeros([ 5, 80 ]).numpy())
	list_label.append(0)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
CHUNK = 1024
RECORD_SECONDS = 1
LONG_STEPS = 100000000000

data_1 = tf.zeros([512, 1]).numpy()
data_2 = tf.zeros([512, 1]).numpy()
data_3 = tf.zeros([512, 1]).numpy()
data_4 = tf.zeros([512, 1]).numpy()
data_frames = [ data_1, data_2, data_3, data_4 ]

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
loggings = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\loggings.log"

checkpoint_clusters_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_Cluster_DataSets_01.h5"
checkpoint_clusters_dir = os.path.dirname(checkpoint_clusters_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)
	
if not exists(checkpoint_clusters_dir) : 
	os.mkdir(checkpoint_clusters_dir)
	print("Create directory: " + checkpoint_clusters_dir)
	
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def animate( i ):
	global CHUNK
	global RATE
	global n_steps
	global data_frames
	
	data = stream.read( CHUNK )
	audio_data = tf.io.decode_raw(tf.constant( data ), tf.int32)
	audio_data = tf.round( audio_data )
	audio_data = tf.cast( audio_data, dtype=tf.float32 )
	
	data_frames = append_data_frames( audio_data, data_frames )
	temp = extract_data_frames( data_frames )

	stfts = tf.signal.stft(temp, frame_length=256, frame_step=64, fft_length=256)
	spectrograms = tf.abs(stfts)
	
	num_spectrogram_bins = stfts.shape[-1]
	lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
	linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins, int( RATE * 2 ), lower_edge_hertz, upper_edge_hertz)
	mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)
	mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate( linear_to_mel_weight_matrix.shape[-1:]) )
	mel_spectrograms = tf.expand_dims(mel_spectrograms, 2)
	
	image = tf.keras.preprocessing.image.array_to_img(
		mel_spectrograms,
		data_format=None,
		scale=True
	)
	
	result = predict_action( mel_spectrograms )
	cluster = predict_cluster( mel_spectrograms )
	im.set_array( image )

	plt.xlabel( str( result + 1 ) + " : " + str( cluster + 1 ), fontsize=22 )
	plt.show()
	
	if n_steps % 8 == 0 :
	
		dataset = tf.data.Dataset.from_tensor_slices((tf.constant([mel_spectrograms.numpy()], shape=(1, 1, 5, 80, 1), dtype=tf.float32),tf.constant([result], shape=(1, 1, 1, 1), dtype=tf.int64)))
		history = model.fit(dataset, epochs=5 ,validation_data=(dataset), callbacks=[custom_callback])
		model.save_weights(checkpoint_path)
	
	if n_steps % 24 == 0 :
	
		dataset = tf.data.Dataset.from_tensor_slices((tf.constant([mel_spectrograms.numpy()], shape=(1, 1, 5, 80, 1), dtype=tf.float32),tf.constant([cluster], shape=(1, 1, 1, 1), dtype=tf.int64)))
		history = model_clusters.fit(dataset, epochs=5 ,validation_data=(dataset), callbacks=[custom_callback])
		model_clusters.save_weights(checkpoint_clusters_path)
		
		
	n_steps = n_steps + 1
	
	return im,

def predict_action( image ) :

	predictions = model.predict(tf.constant(image, shape=(1, 5, 80, 1) , dtype=tf.float32))
	result = tf.math.argmax(predictions[0]).numpy()

	return result

def predict_cluster( image ) :

	predictions = model_clusters.predict(tf.constant(image, shape=(1, 5, 80, 1) , dtype=tf.float32))
	result = tf.math.argmax(predictions[0]).numpy()
	return result

def append_data_frames( data, data_frames ) :	

	data_1 = data_frames[0]
	data_2 = data_frames[1]
	data_3 = data_frames[2]
	data_4 = tf.constant( data, shape=( 512, 1 ) ).numpy()
	data_frames = [ data_1, data_2, data_3, data_4 ]

	return data_frames

def extract_data_frames( data_frames ) :

	temp = tf.stack([data_frames[0], data_frames[1], data_frames[2], data_frames[3]])
	temp = tf.constant( temp, shape=(1, 4 * 512, 1 ))
	predictions = model_frames_extract.predict(temp)
	model_frames_extract.reset_metrics()
	
	temp = tf.constant( predictions[0], shape=( 512, ))

	return temp

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=( 5, 80, 1 )),
	tf.keras.layers.Reshape((5, 80)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(7 * 4))
model.summary()

model_clusters = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=( 5, 80, 1 )),
	tf.keras.layers.Reshape((5, 80)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
])

model_clusters.add(tf.keras.layers.Flatten())
model_clusters.add(tf.keras.layers.Dense(64))
model_clusters.add(tf.keras.layers.Dense(4))
model_clusters.summary()

model_frames_extract = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=( 512 * 4, 1 )),
	tf.keras.layers.Flatten( ),
	tf.keras.layers.Dense(512, activation=tf.nn.softmax),
	])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

fig = plt.figure()
image = plt.imread( "F:\\datasets\\downloads\\cats_name\\train\\Symbols\\01.jpg" )
im = plt.imshow(image)
im.set_cmap('jet')

lable = 0
if im.get_array().shape[0] > 5 :
	dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([1, 1, 5, 80, 1]),tf.constant([lable], shape=(1, 1, 1, 1), dtype=tf.int64)))
else :
	dataset = tf.data.Dataset.from_tensor_slices((tf.constant([im.get_array()], shape=(1, 1, 5, 80, 1), dtype=tf.float32),tf.constant([lable], shape=(1, 1, 1, 1), dtype=tf.int64)))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		if logs['loss'] <= 0.2 :
			self.model.stop_training = True

custom_callback = custom_callback(patience=6)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.Nadam( learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam' )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])
model_frames_extract.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])
model_clusters.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(dataset, epochs=1 ,validation_data=(dataset))

model.save_weights(checkpoint_path)
	
while LONG_STEPS > 0:
	ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
	plt.show()
	
audio.terminate()

print(  "finished recording" )
