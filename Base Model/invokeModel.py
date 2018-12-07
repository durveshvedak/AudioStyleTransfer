import librosa.display
from audio_utils import *
from deepmodel import deeperModel


content_file = "/Users/durveshvedak/Desktop/Fall 2018/Deep Learning/AudioTrainedNetwork/applause2.wav"
style_file = "/Users/durveshvedak/Desktop/Fall 2018/Deep Learning/AudioTrainedNetwork/Results/Shallow CNN/Result 4/laser.wav"

def read_audio_spectogram(filename, N_FFT):
    wav, sampleRate = librosa.load(filename)
    spectrogram = librosa.stft(wav, N_FFT)
    phase = np.angle(spectrogram)
    magnitude = np.abs(spectrogram)
    logMagnitude = np.log1p(magnitude)
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='log', x_axis='time')
    librosa.display.waveplot(wav)

    return logMagnitude, sampleRate


contentSpectro, content_sr = read_audio_spectogram(content_file, N_FFT = 2048)
styleSpectro, style_sr = read_audio_spectogram(style_file, N_FFT = 2048)

N_binsC, N_timestepsC = contentSpectro.shape
N_binsS, N_timestepsS = styleSpectro.shape

print("Content Spectrogram Shape = {}".format(contentSpectro.shape))
print("Style Spectrogram Shape = {}".format(styleSpectro.shape))

styleSpectro = styleSpectro[:N_binsC,:N_timestepsC]

reshapedContent = np.reshape(contentSpectro.T, (1,1,N_timestepsC, N_binsC))
reshapedStyle = np.reshape(styleSpectro.T, (1,1,N_timestepsS, N_binsS))

content_tf = tf.constant(reshapedContent, name = 'content_tf', dtype = tf.float32)
style_tf = tf.constant(reshapedStyle, name = 'style_tf', dtype = tf.float32)

print("Content Tensor {}".format(content_tf))
print("Style Tensor {}".format(style_tf))

usingKickStart = False
model = deeperModel(usingKickStart, content_tf, style_tf, N_binsC)
result = model.synthesize()
writeOutput(result, content_sr, filename = "out.wav")
