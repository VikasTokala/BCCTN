from calendar import c
import librosa as lc
import soundfile as sf
import numpy as np
import os
from pydub import AudioSegment

fs = 16000

filelist_path = '/Users/vtokala/Documents/Research/Databases/Ambient_noise_16k'


def clip(wave, t_length=2, fs=16000):
    segment_length = t_length * fs
    wave_length = wave.shape[0]
    waves = []
    # breakpoint()
    if wave_length < segment_length:
        breakpoint()
       
        waves.append(np.concatenate([wave, np.zeros((2,segment_length - wave_length))]))

    elif wave_length == segment_length:
        waves.append(wave)
    elif wave_length > segment_length:
        num = wave_length // segment_length
        for n in range(num + 1):
            if n < num:
                waves.append(wave[n * segment_length:(n + 1) * segment_length])
            elif n == num:
                waves.append(wave[wave_length - segment_length:])
    return waves


if os.path.isdir(filelist_path + '_clip'):
    pass
else:
    os.mkdir(filelist_path + '_clip')

clean_files = os.listdir(filelist_path)

for idy, clean_filename in enumerate(clean_files):
    print(idy, 'Processing ', clean_filename)
    clean_wave, _ = sf.read(filelist_path + '/' + clean_filename)

    clipped_waves = clip(clean_wave)

    for idx, clean in enumerate(clipped_waves):
        sf.write(filelist_path + '_clip' + '/' + clean_filename[:len(clean_filename) - 4] + '_' + str(idx) + '.wav',
                 clean, fs)
