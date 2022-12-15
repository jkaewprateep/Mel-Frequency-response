# Mel-Frequency-response
Mel frequency response from audio source input

## Wave read ###
Read wave from source
```
data = stream.read( CHUNK )
audio_data = tf.io.decode_raw(tf.constant( data ), tf.int32)
audio_data = tf.round( audio_data )
audio_data = tf.cast( audio_data, dtype=tf.float32 )
	
data_frames = append_data_frames( audio_data, data_frames )
temp = extract_data_frames( data_frames )

stfts = tf.signal.stft(temp, frame_length=256, frame_step=64, fft_length=256)
spectrograms = tf.abs(stfts)
```

## Spectrograms short-time Furrier transfroms ###
Read wave from source
```
stfts = tf.signal.stft(temp, frame_length=256, frame_step=64, fft_length=256)
spectrograms = tf.abs(stfts)
```

## Mel frequency Spectrograms ###
Mel frequency spectrograms, weight matrix
```
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
```

## Target label prediction ###
Model.predict( ) estimate target result as mapping input wave to target Keys
```
def predict_action( image ) :
  
  predictions = model.predict(tf.constant(image, shape=(1, 5, 80, 1) , dtype=tf.float32))
  result = tf.math.argmax(predictions[0]).numpy()
  return result
```

## Result image ##
![Alt text](https://github.com/jkaewprateep/Mel-Frequency-response/blob/main/Figure_2.png?raw=true "Title")
