from typing import Callable, Self
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

# type alias that encapsulates both numbers and functions
WaveParam = float | int | Callable[[float], float]


def import_wav_as_waveform(path: str) -> Callable[[float], float]:
    """
    takes a wav file and interprets it as a waveform function

    to work properly, the wav must contain a single period (oscilation) of the function at 1Hz

    no Fourier-transform is applied, so it will get distorted if the sound is too complex, and will raise an error if it's stereo audio
    """
    _, data = wavfile.read(path)

    if len(data.shape) > 1:
        raise ValueError("The WAV file must be mono (single channel).")

    # transforming in a list[float]
    data = data.astype(float).tolist()
    data = [i / 32767 for i in data]

    # creating the waveform
    def get_val(time: float):
        time %= 1
        index = int(time * len(data))
        return data[index]

    return get_val


# checks WaveParam and let it be called even if it's a number
class WaveParamHandler:
    def __init__(self, content: WaveParam):
        if (
            not isinstance(content, float)
            and not isinstance(content, int)
            and not callable(content)
        ):
            raise TypeError(
                "'content' must be either a float, an int, or a function that takes a float and returns a float."
            )
        self.content = content

    def __call__(self, arg: float):
        if isinstance(self.content, float) or isinstance(self.content, int):
            return self.content
        else:
            return self.content(arg)


class AudioWave:

    def new(
        self,
        freq: WaveParam,
        amp: WaveParam,
        duration: float,
        samplerate: int,
        waveform: Callable[[float], float],
    ):

        freq = WaveParamHandler(freq)
        amp = WaveParamHandler(amp)

        if duration < 0 or samplerate < 0:
            raise ValueError("'duration' and 'samplerate' must be positive numbers.")

        # general initialization of the new fields
        self.wave = []
        self.samplerate = samplerate
        # integer that will store how many voices are present, used for scaling during playing, exporting and appending audio
        self.significance: int = 1
        # stored for convenience
        self.duration = duration

        # wave construction
        if not callable(freq.content):
            # for constant freq, integration is simplified to multiplication
            for time in np.linspace(0, duration, int(samplerate * duration)):
                self.wave.append(waveform(time * freq.content) * amp(time))

        else:
            # if freq is not constant, integration is necessary
            # start is 0
            time = 0
            # dt for the discrete integration
            delta_time = 1 / samplerate
            # cumulative sum of applied frequencies
            csumfreq = 0
            # riemann summation
            while time < duration:
                csumfreq += freq(time) * delta_time
                self.wave.append(waveform(csumfreq) * amp(time))
                time += delta_time

        return self

    @property
    def copy(self):
        """
        "shallow" copy
        """
        aux = AudioWave()
        aux.wave = self.wave.copy()
        aux.samplerate = self.samplerate
        aux.duration = self.duration
        aux.significance = self.significance
        return aux

    def __add__(self, other: Self):
        if other.samplerate != self.samplerate:
            raise ValueError(
                "All waves must have the same samplerate in order to join."
            )

        aux = AudioWave()
        """
        sums element-wise, and considers 0 for out of range indexes
        """
        maxlen = max(len(self.wave), len(other.wave))
        aux.wave = [
            (self.wave[i] if i < len(self.wave) else 0)
            + (other.wave[i] if i < len(other.wave) else 0)
            for i in range(maxlen)
        ]
        aux.significance = self.significance + other.significance
        aux.duration = max(self.duration, other.duration)
        aux.samplerate = self.samplerate
        return aux

    def scale(self, factor: float):
        """
        scales the wave by a given factor, modifying it in place
        """
        self.wave = (np.array(self.wave) * factor).tolist()
        return self

    def __mul__(self, factor: float):
        """
        returns a new AudioWave scaled by a given factor
        """
        copy_wave = self.copy
        copy_wave.scale(factor)
        return copy_wave

    def append(
        self,
        other: Self,
        new_significance: int = 1,
        keep_significance_of_other: bool = False,
    ):
        """
        "appends" another audio to itself

        default significance of the output is 1, because honestly i'm not sure what to do there

        'keep_significance_of_other' is a flag for copying the other wave before appending. This results in preserving the significance of this wave, however might be slower
        """
        if other.samplerate != self.samplerate:
            raise ValueError(
                "All waves must have the same samplerate in order to join."
            )
        self.duration += other.duration

        # scaling to remove differences in amplitude created by adding many waves
        self.scale(1 / self.significance)

        if keep_significance_of_other:
            normalized_other_wave = other * (1 / (other.significance))
            self.wave += normalized_other_wave.wave
        else:
            other.scale(1 / other.significance)
            other.significance = 1
            self.wave += other.wave

        self.significance = new_significance

    @property
    def play(self):
        aux = self * (1 / self.significance)
        sd.play(aux.wave, aux.samplerate)
        sd.wait()

        # in case you want to play it directly after creating
        return self

    def export_wav(self, filename: str):
        aux = self * (1 / self.significance)
        scaled_wave = np.int16(np.array(aux.wave) * 32767)  # scale to 16-bit PCM
        wavfile.write(filename, aux.samplerate, scaled_wave)

        # in case you want to export it directly after creating
        return self
