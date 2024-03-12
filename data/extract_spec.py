import librosa
from os.path import join as join
import numpy as np
import os
import random
from tqdm import tqdm
root_dir = "/data1/zhangxiaohui/CREMA-D/audio/"
sets = ["train", "test"]
for i, source_set in enumerate(sets):
    allwavs = [wav for wav in os.listdir(join(root_dir, source_set)) if wav.endswith(".wav")]
    for wav in tqdm(allwavs):
        wav_path = join(root_dir, source_set, wav)
        sample, rate = librosa.load(wav_path, sr=16000, mono=True)
        while len(sample)/rate < 10.:
            sample = np.tile(sample, 2)
        start_point = random.randint(a=0, b=rate*5)
        new_sample = sample[start_point : start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        if i == 0:
            save_path = join(root_dir, "train_spec", wav.replace(".wav", ".npy"))
        else:
            save_path = join(root_dir, "test_spec", wav.replace(".wav", ".npy"))
        np.save(save_path, spectrogram, allow_pickle = True)