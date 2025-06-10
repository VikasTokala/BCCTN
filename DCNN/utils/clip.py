# audio_clip.py
import librosa.core as lc
import soundfile as sf
import numpy as np
import os

#segment_length = 4
fs = 8000
path = '.'
namelist = [
    [path + '/Speech_train_8k',
     path + '/Noise_train_8k'],
    [path + '/Speech_test_8k',
     path + '/Noise_test_8k'],
]

def clip(wave, t_length=4, fs=8000):
    segment_length = t_length * fs
    wave_length = wave.shape[0]
    waves = []
    if wave_length < segment_length:
        np.pad(wave, (0, segment_length - wave.shape[0]), 'constant', constant_values=0)
    elif wave_length == segment_length:
        waves.append(wave)
    elif wave_length > segment_length:
        num = wave_length // segment_length
        for n in range(num+1):
            if n < num:
                waves.append(wave[n * segment_length:(n + 1) * segment_length])
            elif n == num:
                waves.append(wave[wave_length - segment_length:])
    return waves


for clean_item, noisy_item in namelist:
    if os.path.isdir(clean_item + '_clip'):
        pass
    else:
        os.mkdir(clean_item + '_clip')

    if os.path.isdir(noisy_item + '_clip'):
        pass
    else:
        os.mkdir(noisy_item + '_clip')

    # print('processing' + item)
    clean_files = os.listdir(clean_item)
    noisy_files = os.listdir(noisy_item)

    for clean_filename, noisy_filename in zip(clean_files, noisy_files):
        print(clean_item, ' ', noisy_item, ' ', clean_filename)
        clean_wave, _ = lc.load(clean_item + '/' + clean_filename, sr=fs)
        noisy_wave, _ = lc.load(noisy_item + '/' + noisy_filename, sr=fs)

        clean_waves = clip(clean_wave)
        noisy_waves = clip(noisy_wave)

        for idx, [clean, noisy] in enumerate(zip(clean_waves, noisy_waves)):
            sf.write(clean_item + '_clip' + '/' + clean_filename[:len(clean_filename) - 4] + '_' + str(idx) + '.wav', clean, fs)
            sf.write(noisy_item + '_clip' + '/' + noisy_filename[:len(noisy_filename) - 4] + '_' + str(idx) + '.wav', noisy, fs)