# Mel-Frequency-response
Mel frequency response from audio source input, there are many application of domain transform functions because in nature we senses of some repeating information to understand or its signaturesto recognized. Computer understand it the same way as we build the program but we do not need to spend more time of the repeating information then we transfroms it by extracting important information from the wave sources and transform into new format as data preparation.

Data preparation composed of two or more mainly part which are
> 1. Data cleaning 
> 2. Data extracting or data input information significant

## Input / Output ###
Tensorflow model training and saved data format is easier to manage with picture format and naming that is because we also can see it and understand it easy. One way is create folder management for cataforized of them with mapping labels and description. We can use of the transfromed format to working later or continue working on the next computer deloyment.
| Input  | Output |
| ------------- | ------------- |
| Wave from source  | MFCC as image  |
| Wave files  | MFCC as image  |

## Wave read ###
Read wave from source, we need to select number of input and process to response time and target frequency response by the audio source we create function to action to the input as mono microphone or 1024 bytes, read the input wave format they contain of wave signals and header format. Those parameters of information we extracting are from the header format, couting or mapping method working only within it scope sometime we found records extracts record as wave files failed or unrelated data because they are not create the correct header by it format ( IEEE )
```
data = stream.read( CHUNK )
audio_data = tf.io.decode_raw(tf.constant( data ), tf.int32)
audio_data = tf.round( audio_data )
audio_data = tf.cast( audio_data, dtype=tf.float32 )
	
data_frames = append_data_frames( audio_data, data_frames )
```

## Data Cleanzing ###
Sometimes working with samples from environment sources there are data learning interuption with Furrier transfrom and backward furrier transform can filters out unrelated data or sustain the data continuous.

![Alt text](https://github.com/jkaewprateep/Mel-Frequency-response/blob/main/10.png?raw=true "Title")
![Alt text](https://github.com/jkaewprateep/Mel-Frequency-response/blob/main/4.png?raw=true "Title")
![Alt text](https://github.com/jkaewprateep/Mel-Frequency-response/blob/main/5.png?raw=true "Title")
![Alt text](https://github.com/jkaewprateep/Mel-Frequency-response/blob/main/9.png?raw=true "Title")


## Spectrograms short-time Furrier transfroms ###
Short-time Furrier transfroms, by frame_step or windows step and the attention length generate output from time domain to frequency domain. 
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
