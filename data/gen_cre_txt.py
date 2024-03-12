import os
from random import shuffle
from tqdm import tqdm
class_dict = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}

video_data_dir = "/data1/zhangxiaohui/CREMA-D/VideoFlash/"
audio_data_dir = "/data1/zhangxiaohui/CREMA-D/AudioWAV/"

all_files = os.listdir(video_data_dir)

shuffle(all_files)

test_len = int(len(all_files)*0.1)

test_files = all_files[:test_len]
train_files = all_files[test_len:]

train_info, test_info = [], []

for filename in tqdm(train_files):
    label = class_dict[filename.split("_")[2]]
    source_v_path = os.path.join(video_data_dir, filename)
    source_a_path = os.path.join(audio_data_dir, filename.replace(".flv", ".wav"))
    target_v_path = os.path.join(video_data_dir.replace("VideoFlash", "visual/train/"), filename)
    target_a_path = os.path.join(video_data_dir.replace("VideoFlash", "audio/train/"), filename.replace(".flv", ".wav"))
    train_info.append("{} {}\n".format(filename, label))
    os.system("cp {} {}".format(source_a_path, target_a_path))
    os.system("cp {} {}".format(source_v_path, target_v_path))

for filename in tqdm(test_files):
    label = class_dict[filename.split("_")[2]]
    source_v_path = os.path.join(video_data_dir, filename)
    source_a_path = os.path.join(audio_data_dir, filename.replace(".flv", ".wav"))
    target_v_path = os.path.join(video_data_dir.replace("VideoFlash", "visual/test/"), filename)
    target_a_path = os.path.join(video_data_dir.replace("VideoFlash", "audio/test/"), filename.replace(".flv", ".wav"))
    test_info.append("{} {}\n".format(filename, label))
    os.system("cp {} {}".format(source_a_path, target_a_path))
    os.system("cp {} {}".format(source_v_path, target_v_path))

with open("my_train_cre.txt", "w") as fw:
    fw.writelines(train_info)

with open("my_test_cre.txt", "w") as fw:
    fw.writelines(test_info)







