# Activity-Recognition-Using-Accelerometer-Data
The goal of this project is to classify the actions taken by the user (walking, climbing stairs, and descending stairs) from the 3D accelerometer data.

## Tools Used 
Signal processing:  Python, Numpy, Scipy and Matplotlib
Classifier design:  Tensorflow-Keras 

## Input Data Reading
Input data is read from text files provided. Input data is read in 9 separate lists - climb_x, climb_y, climb_z, descend_x, descend_y, descend_z, walk_x, walk_y and walk_z. This data is then further divided into sublists of length 70% for training and 30% for testing. These data files were read in random order so that the data won’t be overfitted during the training process.

## Preprocessing
### Raw data plots in time domain:
Signal we have been provided has many noisy frequency components. Thus it is necessary to pass the signal through necessary filters and refine the data.
Following plots show the raw training signal data in the time domain. Plots are X, Y, Z acceleration amplitudes of climb action, descending action and walking action respectively.
It can be observed that in case of climb data, most of the signal follows certain pattern except around sample number 6000. After investigation, it was found that this is due to the training dataset which was recorded when accelerometer was worn by the man m1 and number of m1 samples are very less as compared to the female f1. 
Also, no conclusions can be made by just looking at the training data in time domain. Thus to analyze the frequency content of the signal wrt time, a spectrogram function was created. 

### FFT Plots
Following figures shows the FFT plots of the X axis data of climbing, descending and walking actions. Remaining plots can be obtained by running the python code provided with this report.
<>
FFT plots give us a rough idea of the frequency content in the signal. Since, this is a time varying frequency signal, just looking at the FFT plots give us no information about the time dependencies frequencies i.e how frequency content varies with respect to the time. This information can be obtained by the spectral analysis. 

### Spectrogram analysis
In spectrogram, the time varying signal is divided into number of blocks along the time axis. These blocks are also known as ‘windows’. And then, SFFT is applied on each of these blocks. Resulting in the graph, that gives us the frequency distribution at different time steps.
There are many window functions that can be used for this purpose. Following plots show the different spectrograms obtained by using different windows. 
Parameters that affect SFFT:
NFFT = Number of data points in each window frame. It is convention to set this value equal to some power of 2.
noverlap = Number of data points that are allowed to overlap between nth and (n-1)th frame. 

In the python code provided, you can also run the block that plots the spectrogram using the Kaiser window. It is not added in the report because output is visually similar to the gaussian window.  

## Observations made using Spectrogram:
1. Depending on the size of the Fourier analysis window, different levels of frequency/time resolution are achieved.
2. Effect of window length (NFFT): NFFT gives the width of the window in terms of number of samples. The larger the window size, the better frequency resolution you get as you're capturing more of the frequencies, but the time localization is poor. Similarly, the smaller the window size, the better localization you have in time, but you don't get that great of a frequency decomposition. Thus, window length of 128 was used with hamming window, as it gives the best frequency-time response for all the signals.
3. Effect of noverlap (noverlap): In order to ensure good localization of the frequencies, generally some samples from the adjacent windows are overlapped with each other.Less the overlap, frequency band is more discretized. If overlap is increased, frequency bands appear more continuous. This can be verified by changing the value of the noverlap parameter in the code.
In Matlab, the default value of noverlap is 50% of the window length. However, it is None in case of pyplot’s specgram() function. Thus, in the code, noverlap is set to 0.5*NFFT.

## Filter Design
To design the filter, following observations were considered from spectrogram and fft plots:
1. 0Hz frequency content is present in all the data throughout the signal. It can be observed in both FFT (vertical long line on y axis at 0) and Spectrogram (bright yellow line at the bottom). 0Hz frequency introduces DC content in the signal. Thus it is not of any use for the classification purpose. Thus it should be removed.
2. There are no powerful signals above 8Hz in any signal. It was verified by plotting spectrogram and FFT plots for all the signals (Code is provided for this).
By default this function uses Hamming window.

### Parameters Used:
1. A FIR bandpass filter was designed with low cut off frequency = 0.5Hz (To remove 0Hz Signal content) and upper cut off frequency varying according to each signal in the range of 4 to 8Hz. 
2. Numtaps: The number of taps = number of coefficient s = Length of filter in case of FIR filter. The order of the filter is equal to Length of filter-1. Since the signal under consideration has a large number of samples, it was observed that the numtabs=257 gives the best accurate. Also it is known that FIR filter introduces the delay in the signal. This delay can be calculated using the formula:
delay = 0.5 * numtaps

## Classifier Design
There are multiple traditional ML classifier options available for this problem. For eg: SVM(State vector machines), Decision Trees, KNN(K-Nearest Neighbours). However, the classifier is trained by using a single Neural Network. due to the following reasons:
Even though SVM and KNN have very less number of parameters to train, in this case feature length of the data is very small. Hence, tuning the weights of the Neural Network is faster. Thus more accuracy can be achieved without losing the speed.
Once NN is trained, the model can be improved by continuing the training with new data using the transfer learning. It is not possible in case of SVM, KNN or Decision trees.
Neural nets are much better than SVM and KNN for high dimensionality problems. If in the future, we decide to use more features of the signal for classification, it will be easier to work with neural nets. 

### Data Preparation
First training and testing data was split into 70% and 30%.
Both the data was passed through the bandpass filter which we had designed, to get rid of the noisy signals. 
Both training and testing data is normalized. Because, we have three features (x, y, z) with different scales. Normalizing data accelerates the learning process.

### Data Reshaping:
The sensor is reading the data at the rate of 32Hz. Assuming that we want to predict the human activity every 2 seconds, we need to process signal with 32*2 = 64 data points at a time.Also, there will be 3 values (x,y,z) at each instant, hence total data points that network needs to process every 2 seconds is equal to 64*3 = 192
Hence the incoming training and testing data was segmented in the sequences of  length 192 and each sequence was labeled with the correct label of activity.
Also training result was set to be one hot encoded vector [1,0,0] for climbing, [0,1,0] for descending and [0,0,1] for walking.

Shape: Y_train = (nX3) where were each row represents the class probability

The ReLu activation was used because it is the fastest.
And softmax activation was used in the prediction layer to produce the probabilities associated with each class.

## Testing Result
The testing was done on the dataset that was neither used for filter design nor for the training. Assuming that the data is being predicted every 2 secs, the testing data was also divided into sequences of length 192. This data was then passed to the model trained above. 
