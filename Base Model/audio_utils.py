import tensorflow as tf
import numpy as np
import librosa


# this function takes a file path and extracts a short term fourier transform from the linked file
# it returns a 2 dimensional spectrogram, where each row is a frequency bin and each column is slice in time
def read_audio_spectrum(filename, N_FFT):
    # this is a constant for FFT size, affecting the number of frequency bins for fourier transformation

    wav, sampleRate = librosa.load(filename)
    spectrogram = librosa.stft(wav, N_FFT)

    phase = np.angle(spectrogram)
    magnitude = np.abs(spectrogram)

    logMagnitude = np.log1p(magnitude)
    return logMagnitude, sampleRate


# this function takes the result of optimization, which is a fourier transformation
# in the frequency domain, undoes the log calculation taken by read_audio_spectrum, and
# extracts the raw audio and writes it to out.wav
def writeOutput(result, sampleRate, filename):
    result = np.squeeze(result).T
    expResult = np.exp(result) - 1
    invertedOut = librosa.istft(expResult)
    librosa.output.write_wav(filename, invertedOut, sampleRate)


def writeOutput2(result, sampleRate, filename, shape):
    N_Bins, N_Timesteps = shape
    a = np.zeros(shape)
    a[:N_Bins, :] = np.exp(result[0, 0].T) - 1
    N_FFT = 2048

    # This code is supposed to do phase reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    librosa.output.write_wav(filename, x, sampleRate)


def getSavedOutput(meta_file, checkpoint_directory):
    print("in get saved output")

    result = None
    with tf.Session() as session:
        print("in sess")
        saver = tf.train.import_meta_graph(meta_file)
        print("imported meta graph")
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_directory))
        print("restored")

        graph = tf.get_default_graph()
        print("created graph")

        output = graph.get_tensor_by_name("OUT:0")
        print("got output tensor")
        result = session.run(output)
        print("got result")

    return result


def writeListToFile(costs, file_path):
    outFile = open(file_path, "w")
    for cost in costs:
        outFile.write(str(cost) + "\n")

    outFile.close()


def writeParamsToFile(params, file_path):
    outFile = open(file_path, "w")
    for key, value in params.items():
        outFile.write(key + " = " + str(value) + "\n")

    outFile.close()


def loadParamFromTxt(file_path):
    with open(file_path) as f:
        for line in f:
            value = float(line.strip())
            return value


def writeSingleParamToTxt(value, file_path):
    outFile = open(file_path, "w")
    outFile.write(str(value))
    outFile.close()


def loadList(file_name, convertToFloat=False):
    l = None
    with open(file_name) as f:
        if convertToFloat:
            l = [float(line.strip()) for line in f]
        else:
            l = [line.strip() for line in f]
    return l


def loadSavedNumpyArrays(filenames):
    arrays = []
    for filename in filenames:
        loaded = np.load(filename)
        arrays.append(loaded)

    return arrays


def shuffle(X, Y):
    assert X.shape[0] == Y.shape[0]

    N_rows = X.shape[0]

    permutation = np.random.permutation(np.arange(N_rows))

    shuffledX = X[permutation, :, :, :]
    shuffledY = Y[permutation, :]

    return shuffledX, shuffledY


## Taken and slightly modified from deeplearning.ai 'convolutional model: application' assignment
# Assumes X and Y already shuffled
def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Partition (X, Y). Minus the end case.
    num_complete_minibatches = int(np.floor(m / mini_batch_size))
    # number of mini batches of size mini_batch_size in your partitioning

    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches