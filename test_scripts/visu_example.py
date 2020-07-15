import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioAnalysis
from pyAudioAnalysis import audioVisualization
from python_speech_features import mfcc
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import PIL
import matplotlib.mlab as mlab

audioFile = "C:/Users/jbacle/Music/Michael Jackson - Beat It (Official Video).mp3"
audioFileShort = "C:/Users/jbacle/Music/Michael Jackson - Beat It (Official Video) Extract.mp3"
audioFolder = "C:/Users/jbacle/Music"
audioFileTest = "//titan/1000-RnD/030-AUDIO/20-Evaluation_Protocol/20-Playlists/Objective/p50_f.wav"


def featureExtract(x, f):

    frameSize = 0.1

    # Feature Extraction
    F, f_names = ShortTermFeatures.feature_extraction(x, f, frameSize*f, frameSize*f)

    plt.subplot(2, 1, 1)
    plt.plot(F[0, :])
    plt.xlabel('Frame no')
    plt.ylabel(f_names[0])
    plt.subplot(2, 1, 2)
    plt.plot(F[1, :])
    plt.xlabel('Frame no')
    plt.ylabel(f_names[1])
    plt.show()


# Chromagraph
def chroma(x, f):
    # audioAnalysis.fileChromagramWrapper(audioPath)
    frameSize = 0.1
    specgram, TimeAxis, FreqAxis = ShortTermFeatures.chromagram(
        x, fs, round(fs * frameSize), round(fs * frameSize), True)


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = scipy.ndimage.generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = scipy.ndimage.filters.maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = scipy.ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return PIL.Image.frombytes("RGBA", (w, h), buf.tostring())


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


# FROM dejavu https://github.com/worldveil/dejavu/blob/master/dejavu/logic/fingerprint.py
DEFAULT_AMP_MIN = 10
CONNECTIVITY_MASK = 2
PEAK_NEIGHBORHOOD_SIZE = 10
DEFAULT_FS = 44100
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 5


def get_2D_peaks(arr2D: np.array, plot: bool = False, amp_min: int = DEFAULT_AMP_MIN):
    """
    Extract maximum peaks from the spectogram matrix (arr2D).
    :param arr2D: matrix representing the spectogram.
    :param plot: for plotting the results.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    # Original code from the repo is using a morphology mask that does not consider diagonal elements
    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
    # is to change from the current diamond figure to a just a normal square one:
    #       F   T   F           T   T   T
    #       T   T   T   ==>     T   T   T
    #       F   T   F           T   T   T
    # In my local tests time performance of the square mask was ~3 times faster
    # respect to the diamond one, without hurting accuracy of the predictions.
    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
    # That being said, we generate the mask by using the following function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    struct = scipy.ndimage.morphology.generate_binary_structure(2, CONNECTIVITY_MASK)

    #  And then we apply dilation using the following function
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
    #  change it by the following code:
    #  neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    neighborhood = scipy.ndimage.morphology.iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter mask
    local_max = scipy.ndimage.filters.maximum_filter(arr2D, footprint=neighborhood) == arr2D

    # Applying erosion, the dejavu documentation does not talk about this step.
    background = (arr2D == 0)
    eroded_background = scipy.ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.set_yscale('log')
        ax.scatter(times_filter, freqs_filter)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return list(zip(freqs_filter, times_filter))


def fingerprint(channel_samples: list,
                Fs: int = DEFAULT_FS,
                wsize: int = DEFAULT_WINDOW_SIZE,
                wratio: float = DEFAULT_OVERLAP_RATIO,
                # fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN*4):
    """
    FFT the channel, log transform output, find local maxima, then return locally sensitive hashes.
    :param channel_samples: channel samples to fingerprint.
    :param Fs: audio sampling rate.
    :param wsize: FFT windows size.
    :param wratio: ratio by which each sequential window overlaps the last and the next window.
    # :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list of hashes with their corresponding offsets.
    """
    # FFT the signal and extract frequency components
    arr2D, f, t = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))

    # Apply log transform since specgram function returns linear array. 0s are excluded to avoid np warning.
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))

    local_maxima = get_2D_peaks(arr2D, plot=False, amp_min=amp_min)

    local_maxima_scale = []
    for _ in local_maxima:
        local_maxima_scale.append((f[_[0]], t[_[1]]))

    return local_maxima_scale


def plotSpectro(S, t, f):
    fig, ax = plt.subplots(dpi=200, figsize=(8, 4), constrained_layout=True)
    ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.set_ylim([80.0, 20000.0])
    # S = 10 * np.log10(S)
    vmin = 0
    vmax = 0.1
    t = np.array(t)
    f = np.array(f)
    S = np.transpose(S)
    # ax.pcolormesh(t, f, S, cmap="gnuplot")
    spectro = ax.pcolormesh(t, f, S, cmap="Greys", vmin=vmin, vmax=vmax, shading='gouraud')  # noqa F481
    return ax


def spectro(x, f):
    frameSize = .1
    S, t, f = ShortTermFeatures.spectrogram(
        x, fs, round(fs * frameSize), round(fs * frameSize),
        plot=False,
        show_progress=True
        )
    ax = plotSpectro(S, t, f)
    fingers = fingerprint(x, fs)
    ax.plot([_[1] for _ in fingers], [_[0] for _ in fingers], 'o',
            markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1)
    plt.show()


def featureVisu(folder):
    audioVisualization.visualizeFeaturesFolder(folder, "pca", "")


def thumbnailing(file):
    audioAnalysis.thumbnailWrapper(file, 1)


if __name__ == "__main__":
    fs, x = audioBasicIO.read_audio_file(audioFileTest)
    x = audioBasicIO.stereo_to_mono(x)
    # chroma(x, fs)
    # featureVisu(audioFolder)
    # thumbnailing(audioFile)
    spectro(x, fs)
