import os
from extract_fbank import wav2fbank
import numpy as np
import transformers
import torch
from video_preprocessing import VGGSound_dataset
from tqdm import tqdm
data_dir = "/data1/zhangxiaohui/IEMOCAP/"
all_video_dir = "/data1/zhangxiaohui/IEMOCAP/subvideo/"
all_audio_dir = "/data1/zhangxiaohui/IEMOCAP/subaudio/"

# train_txt, dev_txt, test_txt = [], [], []

def process():
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    with open("my_train_iemo.txt", "r") as fr:
        all_info = fr.readlines()

    for line in tqdm(all_info):
        path, caption, label = line.strip().split(" [split|sign] ")
        filename = path.replace(".mp4", ".wav")
        source_audio_path = os.path.join(all_audio_dir, filename)
        # des_audio_path = os.path.join(data_dir, "audio/train/", filename)
        # os.system("cp {} {}".format(source_audio_path, des_audio_path))
        fbank = wav2fbank(source_audio_path, None, 0)
        save_path = os.path.join(data_dir, "audio/train_fbank", filename.replace(".wav", ".npy"))
        np.save(save_path, fbank, allow_pickle = True)

        encoded_caption = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="np",
            add_special_tokens=False,
        )

        tokenized_caption = torch.from_numpy(encoded_caption["input_ids"][0])[None, ...]
        padding_mask = 1.0 - encoded_caption["attention_mask"][0].astype(np.float32)
        padding_mask = torch.from_numpy(padding_mask[None, ...])

        token_save_path = os.path.join(data_dir, "text_token/train_token", "{}_token.npy".format(filename.split(".wav")[0]))
        pm_save_path = os.path.join(data_dir, "text_token/train_token", "{}_pm.npy".format(filename.split(".wav")[0]))

        np.save(token_save_path, np.array(tokenized_caption))
        np.save(pm_save_path, np.array(padding_mask))

        # source_video_path = os.path.join(all_video_dir, filename.replace(".wav", ".mp4"))
        # des_video_path = os.path.join(data_dir, "visual/train", filename.replace(".wav", ".mp4"))

        # os.system("cp {} {}".format(source_video_path, des_video_path))

    with open("my_dev_iemo.txt", "r") as fr:
        all_info = fr.readlines()
    for line in tqdm(all_info):
        path, caption, label = line.strip().split(" [split|sign] ")
        filename = path.replace(".mp4", ".wav")
        source_audio_path = os.path.join(all_audio_dir, filename)
        # des_audio_path = os.path.join(data_dir, "audio/dev/", filename)
        # os.system("cp {} {}".format(source_audio_path, des_audio_path))
        fbank = wav2fbank(source_audio_path, None, 0)
        save_path = os.path.join(data_dir, "audio/dev_fbank", filename.replace(".wav", ".npy"))
        np.save(save_path, fbank, allow_pickle = True)

        encoded_caption = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="np",
            add_special_tokens=False,
        )

        tokenized_caption = torch.from_numpy(encoded_caption["input_ids"][0])[None, ...]
        padding_mask = 1.0 - encoded_caption["attention_mask"][0].astype(np.float32)
        padding_mask = torch.from_numpy(padding_mask[None, ...])

        token_save_path = os.path.join(data_dir, "text_token/dev_token", "{}_token.npy".format(filename.split(".wav")[0]))
        pm_save_path = os.path.join(data_dir, "text_token/dev_token", "{}_pm.npy".format(filename.split(".wav")[0]))

        np.save(token_save_path, np.array(tokenized_caption))
        np.save(pm_save_path, np.array(padding_mask))

        # source_video_path = os.path.join(all_video_dir, filename.replace(".wav", ".mp4"))
        # des_video_path = os.path.join(data_dir, "visual/dev", filename.replace(".wav", ".mp4"))

        # os.system("cp {} {}".format(source_video_path, des_video_path))
    
    with open("my_test_iemo.txt", "r") as fr:
        all_info = fr.readlines()
    for line in tqdm(all_info):
        path, caption, label = line.strip().split(" [split|sign] ")
        filename = path.replace(".mp4", ".wav")
        source_audio_path = os.path.join(all_audio_dir, filename)
        des_audio_path = os.path.join(data_dir, "audio/test/", filename)
        os.system("cp {} {}".format(source_audio_path, des_audio_path))
        fbank = wav2fbank(source_audio_path, None, 0)
        save_path = os.path.join(data_dir, "audio/test_fbank", filename.replace(".wav", ".npy"))
        np.save(save_path, fbank, allow_pickle = True)

        encoded_caption = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="np",
            add_special_tokens=False,
        )

        tokenized_caption = torch.from_numpy(encoded_caption["input_ids"][0])[None, ...]
        padding_mask = 1.0 - encoded_caption["attention_mask"][0].astype(np.float32)
        padding_mask = torch.from_numpy(padding_mask[None, ...])

        token_save_path = os.path.join(data_dir, "text_token/test_token", "{}_token.npy".format(filename.split(".wav")[0]))
        pm_save_path = os.path.join(data_dir, "text_token/test_token", "{}_pm.npy".format(filename.split(".wav")[0]))

        np.save(token_save_path, np.array(tokenized_caption))
        np.save(pm_save_path, np.array(padding_mask))

        # test_txt.append("{} {}".format(filename.replace(".wav", ".mp4"), label))

        # source_video_path = os.path.join(all_video_dir, filename.replace(".wav", ".mp4"))
        # des_video_path = os.path.join(data_dir, "visual/test", filename.replace(".wav", ".mp4"))

        # os.system("cp {} {}".format(source_video_path, des_video_path))
    
def extract_img():
    vggtrain = VGGSound_dataset('/data1/zhangxiaohui/IEMOCAP/visual/', mode = "train")
    vggtrain.extractImage()
    vggdev = VGGSound_dataset('/data1/zhangxiaohui/IEMOCAP/visual/', mode = "dev")
    vggdev.extractImage()
    vggtest = VGGSound_dataset('/data1/zhangxiaohui/IEMOCAP/visual/', mode = "test")
    vggtest.extractImage()

from PIL import Image
def process_img():
    face_dir = "/data1/zhangxiaohui/IEMOCAP/faces/"
    target_dir = "/data1/zhangxiaohui/IEMOCAP/visual/"
    with open("my_train_iemo.txt", "r") as mit:
        info = mit.readlines()
    for line in tqdm(info):
        filename = line.strip().split()[0].split(".mp4")[0]
        data_dir = os.path.join(face_dir, filename)
        all_images = os.listdir(data_dir)
        for ori_img in all_images:
            s_path = os.path.join(data_dir, ori_img)
            d_path = os.path.join(target_dir, "train_imgs", filename, ori_img)
            if not os.path.exists(os.path.join(target_dir, "train_imgs", filename)):
                os.mkdir(os.path.join(target_dir, "train_imgs", filename))
            img = Image.open(s_path)
            new_img = img.resize((256,256), Image.BILINEAR)
            new_img.save(d_path)
    with open("my_dev_iemo.txt", "r") as mit:
        info = mit.readlines()
    for line in tqdm(info):
        filename = line.strip().split()[0].split(".mp4")[0]
        data_dir = os.path.join(face_dir, filename)
        all_images = os.listdir(data_dir)
        for ori_img in all_images:
            s_path = os.path.join(data_dir, ori_img)
            d_path = os.path.join(target_dir, "dev_imgs", filename, ori_img)
            if not os.path.exists(os.path.join(target_dir, "dev_imgs", filename)):
                os.mkdir(os.path.join(target_dir, "dev_imgs", filename))
            img = Image.open(s_path)
            new_img = img.resize((256,256), Image.BILINEAR)
            new_img.save(d_path)
    with open("my_test_iemo.txt", "r") as mit:
        info = mit.readlines()
    for line in tqdm(info):
        filename = line.strip().split()[0].split(".mp4")[0]
        data_dir = os.path.join(face_dir, filename)
        all_images = os.listdir(data_dir)
        for ori_img in all_images:
            s_path = os.path.join(data_dir, ori_img)
            d_path = os.path.join(target_dir, "test_imgs", filename, ori_img)
            if not os.path.exists(os.path.join(target_dir, "test_imgs", filename)):
                os.mkdir(os.path.join(target_dir, "test_imgs", filename))
            img = Image.open(s_path)
            new_img = img.resize((256,256), Image.BILINEAR)
            new_img.save(d_path)


    

if __name__ == "__main__":
    process()

    process_img()







        
