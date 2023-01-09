import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import jenkspy
import json
import os.path

THRESHOLD = 800


def plot(x, y, opts):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xscale('log', base=2)
    plt.grid()
    plt.xlabel(opts['x'])
    plt.ylabel(opts['y'])
    plt.title(opts['title'])


# load database
with open('./raga.json') as db:
    database = json.load(db)

# read audio file
file = input('Enter file name: ')
if(not(os.path.exists(f'./audio/wav/{file}.wav'))):
    raise FileExistsError('File not found!')
samples, fs = librosa.load(f'./audio/wav/{file}.wav', sr=None)

D = librosa.stft(samples)
S = np.abs(D)
S_db = librosa.amplitude_to_db(S, ref=np.max)

plt.figure()
librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=fs)
plt.colorbar()
plt.title(f'Power spectrogram ({file}.wav)')
plt.xlabel('time')
plt.ylabel('Frequency (log scale)')
plt.show(block=False)

CLUSTER_SIZE = int(input('Enter the count of unique frequencies visible: '))
BAND_PASS_FL = int(input('Enter frequency of madhyama sthayi Sa: '))
BAND_PASS_FH = 2*BAND_PASS_FL

# generate array of frequencies
n = len(samples)
xf = np.linspace(0, int(fs/2), int(n/2))

# compute fft
yf = np.fft.fft(samples)

# plot spectrum before filtering
plot(xf, 2/n*np.abs(yf[:n//2]), {
    'x': 'Frequency',
    'y': 'Magnitude',
    'title': f'Magnitude Spectrum (File: {file}.wav)'
})

# bandpass filter
indices = np.where((xf > BAND_PASS_FL) & (xf < BAND_PASS_FH))
xf = np.take(xf, indices)[0]
yf = np.take(yf, indices)[0]

# plot spectrum after bandpass filtering
plot(xf, 2/n*np.abs(yf[:n//2]), {
    'x': 'Frequency',
    'y': 'Magnitude',
    'title': f'Magnitude Spectrum after Bandpass Filtering (File: {file}.wav)'
})

# zero out amplitudes below threshold
yf = np.where(abs(yf) < THRESHOLD, 0, yf)

# zero out frequencies corresponding to amplitudes below threshold
freqs = np.copy(xf)
for i in range(len(freqs)):
    if(abs(yf[i]) < THRESHOLD):
        freqs[i] = 0

# filter out zeros
amps = np.copy(yf)
freqs = np.array(list(filter(lambda f: (f != 0), freqs)))
amps = np.array(list(filter(lambda a: (a != 0), amps)))

# plot spectrum after threshold noise filtering
plot(xf, 2/n*np.abs(yf[:n//2]), {
    'x': 'Frequency',
    'y': 'Magnitude',
    'title': f'Magnitude Spectrum after Threshold Noise Filtering (File: {file}.wav)'
})

# split frequencies into clusters
clusters = jenkspy.jenks_breaks(freqs, n_classes=CLUSTER_SIZE)
cluster_idx = np.array(
    list(map(lambda bound: (np.where(freqs == bound)[0][0]), clusters))
)
print('Cluster bounding frequencies:', clusters)

# find peaks
peakFreqs = []
for i in range(len(cluster_idx)-1):
    start = cluster_idx[i] if(i == 0) else cluster_idx[i]+1
    stop = cluster_idx[i+1]+1
    peak = freqs[np.where(abs(amps) == max(abs(amps[start:stop])))[0][0]]
    peakFreqs.append(peak)

print('Peak Frequencies:', peakFreqs)

# normalize peaks
normPeakFreqs = [freq/min(peakFreqs) for freq in peakFreqs]
print('Normalized Peak Frequencies:', normPeakFreqs)

# compute euclidean distance
dists = []
for entry in database:
    trueRatios = np.array(entry['ratios'])
    computedRatios = np.array(normPeakFreqs)
    maxLen = max(len(trueRatios), len(computedRatios))
    trueRatios = np.pad(trueRatios, (0, maxLen - len(trueRatios)))
    computedRatios = np.pad(computedRatios, (0, maxLen - len(computedRatios)))
    dist = np.linalg.norm(computedRatios - trueRatios)
    dists.append(dist)

# predict raaga using index of minDist
minDist = min(dists)
minDistIdx = dists.index(minDist)
print('The raaga is:', database[minDistIdx]['raaga'])
print('Euclidean distance:', minDist)

# compute confidence %
trueRatios = database[minDistIdx]['ratios']
mean_t = np.mean(trueRatios)
var_t = np.var(trueRatios)
mean_c = np.mean(normPeakFreqs)
var_c = np.var(normPeakFreqs)
d = len(trueRatios)
maxDist = np.sqrt(d*((mean_c - mean_t)**2 + (var_c + var_t)**2))
error = (minDist/maxDist) * 100
print(f'Confidence = {100 - error}%')

# show all plots
plt.show()
