import copy
import csv
import os
import pickle
import numpy as np

import torch
from PIL import Image
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import torchaudio
from timm.data import create_transform
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint

class AVDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.args = args

        if args.dataset == "KineticSound":
            self.data_root = '/data1/zhangxiaohui/k400/'
            self.visual_feature_path = os.path.join(self.data_root, "kinsound/visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "kinsound/audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_ks.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_ks.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_ks.txt"
        elif args.dataset == "AVE":
            self.data_root = '/data1/zhangxiaohui/AVE_Dataset/'
            self.visual_feature_path = os.path.join(self.data_root, "AVE/visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "AVE/audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_ave.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_ave.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_ave.txt"
        elif args.dataset == "RAVDESS":
            self.data_root = '/data1/zhangxiaohui/RAVDESS/'
            self.visual_feature_path = os.path.join(self.data_root, "visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_rav.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_rav.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_rav.txt"
        elif args.dataset == "CREMAD":
            self.data_root = '/data1/zhangxiaohui/CREMA-D/'
            self.visual_feature_path = os.path.join(self.data_root, "visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_cre.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_cre.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_cre.txt"


        # with open(self.stat_path) as f1:
        #     csv_reader = csv.reader(f1)
        #     for row in csv_reader:
        #         classes.append(row[0])
        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        if args.dataset == "KineticSound":
            classes = [sclass.strip().split(" >")[0] for sclass in classes if len(sclass.strip().split(" >")) == 3]
        else:
            classes = [sclass.strip() for sclass in classes]
        # assert len(classes) == 23

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            # csv_reader = csv.reader(f2)
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                if args.dataset == "CREMAD":
                    item = single_line.strip().split(".flv ")
                else:
                    item = single_line.strip().split(".mp4 ")
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.npy')
                visual_path = os.path.join(self.visual_feature_path, item[0])
                # pdb.set_trace()
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    continue

        self.classes = sorted(classes)

        # print(self.classes)
        self.data2class = data2class
        self.av_files = []
        for item in data:
            self.av_files.append(item)
        if self.args.modulation == "QMF" and self.args.mask_percent > 0 and mode == "train":
            mask_start = int(len(self.av_files)*(1 - self.args.mask_percent))
            self.mask_av_files = self.av_files[mask_start:]
            print('# of masked files = %d ' % len(self.mask_av_files))
        else:
            self.mask_av_files = []
            print('# of masked files = %d ' % len(self.mask_av_files))
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.npy')
        # spectrogram = pickle.load(open(audio_path, 'rb'))
        spectrogram = np.load(audio_path)
        if self.args.modulation == "QMF" and av_file in self.mask_av_files and self.args.mask_m == "audio":
            spectrogram = spectrogram * 0
            # print("mask audio {}".format(spectrogram))

        # Visual
        visual_path = os.path.join(self.visual_feature_path, av_file)
        allimages = os.listdir(visual_path)
        file_num = len(allimages)

        if self.mode == 'train':

            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        seg = int(file_num / pick_num)
        image_arr = []

        for i in range(pick_num):
            tmp_index = int(seg * i)
            image = Image.open(os.path.join(visual_path, allimages[tmp_index])).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(1).float()
            image_arr.append(image)
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)
        if self.args.modulation == "QMF" and av_file in self.mask_av_files and self.args.mask_m == "visual":
            image_n = image_n * 0
            # print("mask visual {}".format(image_n))
        label = self.classes.index(self.data2class[av_file])

        return spectrogram, image_n, label, torch.LongTensor([idx])

class CAVDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.augnois = args.cav_augnois
        if args.dataset == "KineticSound":
            self.data_root = '/data1/zhangxiaohui/k400/'
            self.visual_feature_path = os.path.join(self.data_root, "kinsound/visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "kinsound/audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_ks.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_ks.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_ks.txt"
        elif args.dataset == "AVE":
            self.data_root = '/data1/zhangxiaohui/AVE_Dataset/'
            self.visual_feature_path = os.path.join(self.data_root, "AVE/visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "AVE/audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_ave.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_ave.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_ave.txt"
        elif args.dataset == "RAVDESS":
            self.data_root = '/data1/zhangxiaohui/RAVDESS/'
            self.visual_feature_path = os.path.join(self.data_root, "visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_rav.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_rav.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_rav.txt"
        elif args.dataset == "CREMAD":
            self.data_root = '/data1/zhangxiaohui/CREMA-D/'
            self.visual_feature_path = os.path.join(self.data_root, "visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_cre.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_cre.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_cre.txt"

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        if args.dataset == "KineticSound":
            classes = [sclass.strip().split(" >")[0] for sclass in classes if len(sclass.strip().split(" >")) == 3]
        else:
            classes = [sclass.strip() for sclass in classes]

        # assert len(classes) == 23
        # import pdb
        # pdb.set_trace()

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            # csv_reader = csv.reader(f2)
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                if args.dataset == "CREMAD":
                    item = single_line.strip().split(".flv ")
                else:
                    item = single_line.strip().split(".mp4 ")
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.npy')
                visual_path = os.path.join(self.visual_feature_path, item[0])
                # pdb.set_trace()
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    # if args.dataset == 'AVE':
                    #     # AVE, delete repeated labels
                    #     a = set(data)
                    #     if item[1] in a:
                    #         del data2class[item[1]]
                    #         data.remove(item[1])
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class
        # import pdb
        # pdb.set_trace()

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4850, 0.4560, 0.4060],std=[0.2290, 0.2240, 0.2250])
            ])
        self.skip_norm = False
        self.noise = True
        self.norm_mean = -5.081
        self.norm_std = 4.4849
        

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def fbank_aug(self, feature, freqm_m = 48, timem_m = 192):
        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(freqm_m)
        timem = torchaudio.transforms.TimeMasking(timem_m)
        fbank = torch.transpose(feature, 0, 1)
        fbank = fbank.unsqueeze(0)
        if freqm_m != 0:
            fbank = freqm(fbank)
        if timem_m != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        return fbank

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.npy')
        fbank = np.load(audio_path)
        fbank = torch.tensor(fbank)
        if self.mode == "train" and self.augnois:
            fbank = self.fbank_aug(fbank)

        # Visual
        visual_path = os.path.join(self.visual_feature_path, av_file)
        allimages = os.listdir(visual_path)
        file_num = len(allimages)
        image = self.get_image(os.path.join(visual_path, allimages[int(file_num / 2)]))

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True and self.mode == "train" and self.augnois:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-1024, 1024), 0)

        label = self.classes.index(self.data2class[av_file])
        
        return fbank, image, label
    
class M3AEDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.augnois = args.cav_augnois
        self.dataset = args.dataset
        if args.dataset == "Food101":
            self.data_root = '/data1/zhangxiaohui/food101/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token", '{}_token/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_food.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_food.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_food.txt"
        elif args.dataset == "MVSA":
            self.data_root = '/data1/zhangxiaohui/MVSA_Single/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token", '{}_token/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_mvsa.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_mvsa.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_mvsa.txt"
        elif args.dataset == "CUB":
            self.data_root = '/data1/zhangxiaohui/CUB_200_2011/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token", '{}_token/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_cub.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_cub.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_cub.txt"

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(".jpg ")
                token_path = os.path.join(self.text_feature_path, item[0] + '_token.npy')
                pm_path = os.path.join(self.text_feature_path, item[0] + '_pm.npy')
                if args.dataset == "MVSA" or args.dataset == "Food101" or args.dataset == "CUB":
                    visual_path = os.path.join(self.visual_feature_path, item[0] + ".jpg")    
                else:
                    visual_path = os.path.join(self.visual_feature_path, item[0])
                # pdb.set_trace()
                if os.path.exists(token_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))
        # self.preprocess = transforms.Compose([
        #     transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4850, 0.4560, 0.4060],std=[0.2290, 0.2240, 0.2250])
        #     ])
        self.preprocess_train = create_transform(
                input_size = 256,
                is_training=True,
                color_jitter = True,
                auto_augment = None,
                interpolation = "bicubic",
                re_prob = 0,
                re_mode = 0,
                re_count = "const",
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225),
            )
        self.preprocess_test = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        self.skip_norm = True
        self.noise = False
        self.norm_mean = -5.081
        self.norm_std = 4.4849
        

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            if self.mode == "train":
                image_tensor = self.preprocess_train(img)
            else:
                image_tensor = self.preprocess_test(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        pm_path = os.path.join(self.text_feature_path, av_file + '_pm.npy')
        tokenizer = np.load(token_path)
        padding_mask = np.load(pm_path)
        tokenizer = torch.tensor(tokenizer)
        padding_mask = torch.tensor(padding_mask)

        # Visual
        if self.dataset == "MVSA" or self.dataset == "Food101" or self.dataset == "CUB":
            image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        else:
            visual_path = os.path.join(self.visual_feature_path, av_file)
            allimages = os.listdir(visual_path)
            file_num = len(allimages)
            image = self.get_image(os.path.join(visual_path, allimages[int(file_num / 2)]))
        # normalize the input for both training and test
        if self.skip_norm == False:
            tokenizer = (tokenizer - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True and self.mode == "train" and self.augnois:
            tokenizer = tokenizer + torch.rand(tokenizer.shape[0], tokenizer.shape[1]) * np.random.rand() / 10
            tokenizer = torch.roll(tokenizer, np.random.randint(-1024, 1024), 0)

        label = self.classes.index(self.data2class[av_file])
        
        return tokenizer, padding_mask, image, label, torch.LongTensor([idx])
    
class TVDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode

        if args.dataset == "KineticSound":
            self.data_root = '/data1/zhangxiaohui/k400/'
            self.visual_feature_path = os.path.join(self.data_root, "kinsound/visual/", '{}_imgs/Image-01-FPS/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "kinsound/audio/", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_ks.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_ks.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_ks.txt"
        elif args.dataset == "MVSA":
            self.data_root = '/data1/zhangxiaohui/MVSA_Single/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token", '{}_token/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_mvsa.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_mvsa.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_mvsa.txt"


        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(".jpg ")
                token_path = os.path.join(self.text_feature_path, item[0] + '_token.npy')
                pm_path = os.path.join(self.text_feature_path, item[0] + '_pm.npy')
                if args.dataset == "MVSA" or args.dataset == "Food101" or args.dataset == "CUB":
                    visual_path = os.path.join(self.visual_feature_path, item[0] + ".jpg")    
                else:
                    visual_path = os.path.join(self.visual_feature_path, item[0])
                # pdb.set_trace()
                if os.path.exists(token_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

        self.preprocess_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.preprocess_test = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.av_files)

    def get_image(self, filename, mix_lambda=1):
        img_arr = []
        for i in range(3):
            img = Image.open(filename).convert("RGB")
            if self.mode == "train":
                image_tensor = self.preprocess_train(img)
            else:
                image_tensor = self.preprocess_test(img)
            image_tensor = image_tensor.unsqueeze(1).float()
            img_arr.append(image_tensor)
            if i == 0:
                image_n = copy.copy(img_arr[i])
            else:
                image_n = torch.cat((image_n, img_arr[i]), 1)
        
        return image_n

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        pm_path = os.path.join(self.text_feature_path, av_file + '_pm.npy')
        tokenizer = np.load(token_path)
        padding_mask = np.load(pm_path)
        tokenizer = torch.tensor(tokenizer)
        padding_mask = torch.tensor(padding_mask)

        # Visual
        image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))

        label = self.classes.index(self.data2class[av_file])
        
        return tokenizer, image, label


## copy from cpm-net
def random_mask(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return: Sn [alldata_len, view_num]
    """
    # print (f'==== generate random mask ====')
    one_rate = 1-missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum=2] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num)) # [samplenum, viewnum=2] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        # print (f'first ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        # print (f'second ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        # print (f'third ratio: {ratio}')
        error = abs(one_rate - ratio)
        
    return matrix

class Modal3Dataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        # self.augnois = args.cav_augnois
        self.dataset = args.dataset
        if args.dataset == "IEMOCAP":
            self.data_root = '/data1/zhangxiaohui/IEMOCAP/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token", '{}_token/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "audio", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_iemo.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_iemo.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_iemo.txt"

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(" [split|sign] ")
                item[0] = item[0].split(".mp4")[0]
                token_path = os.path.join(self.text_feature_path, item[0] + '_token.npy')
                pm_path = os.path.join(self.text_feature_path, item[0] + '_pm.npy')
                visual_path = os.path.join(self.visual_feature_path, item[0])
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.npy')
                # pdb.set_trace()
                if os.path.exists(token_path) and os.path.exists(visual_path) and os.path.exists(audio_path):
                    data.append(item[0])
                    data2class[item[0]] = item[-1]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

        self.preprocess_train = create_transform(
                input_size = 256,
                is_training=True,
                color_jitter = True,
                auto_augment = None,
                interpolation = "bicubic",
                re_prob = 0,
                re_mode = 0,
                re_count = "const",
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225),
            )
        self.preprocess_test = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.norm_mean = -5.081
        self.norm_std = 4.4849

        if args.mask_percent or args.mask_percent == 0:
            samplenum = len(self.av_files)
            # print (f'using random initialized mask!!')
            # acoustic_mask = (np.random.rand(samplenum, 1) > self.mask_rate).astype(int)
            # vision_mask = (np.random.rand(samplenum, 1) > self.mask_rate).astype(int)
            # lexical_mask = (np.random.rand(samplenum, 1) > self.mask_rate).astype(int)
            # self.maskmatrix = np.concatenate((acoustic_mask, vision_mask, lexical_mask), axis=1)
            self.maskmatrix = random_mask(3, samplenum, args.mask_percent) # [samplenum, view_num]
            # pdb.set_trace()
        

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            if self.mode == "train":
                image_tensor = self.preprocess_train(img)
            else:
                image_tensor = self.preprocess_test(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        pm_path = os.path.join(self.text_feature_path, av_file + '_pm.npy')
        tokenizer = np.load(token_path)
        padding_mask = np.load(pm_path)
        tokenizer = torch.tensor(tokenizer)
        padding_mask = torch.tensor(padding_mask)

        # Visual
        # image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        # normalize the input for both training and test
        visual_path = os.path.join(self.visual_feature_path, av_file)
        allimages = os.listdir(visual_path)
        image = self.get_image(os.path.join(visual_path, allimages[int(len(allimages) / 2)]))
        # file_num = len(allimages)
        # pick_num = 1
        # seg = int(file_num / pick_num)
        # image_arr = []

        # for i in range(pick_num):
        #     tmp_index = int(seg * i / 2)
        #     # image = Image.open(os.path.join(visual_path, allimages[tmp_index])).convert('RGB')
        #     # image = transform(image)
        #     image = self.get_image(os.path.join(visual_path, allimages[tmp_index]))
        #     image = image.unsqueeze(1).float()
        #     image_arr.append(image)
        #     if i == 0:
        #         image_n = copy.copy(image_arr[i])
        #     else:
        #         image_n = torch.cat((image_n, image_arr[i]), 1)
        
        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.npy')
        spectrogram = np.load(audio_path)
        spectrogram = torch.tensor(spectrogram)

        label = self.classes.index(self.data2class[av_file])

        mask_seq = self.maskmatrix[idx]
        missing_index = torch.LongTensor(mask_seq)
        # print(missing_index, missing_index.shape)
        # pdb.set_trace()
        spectrogram = spectrogram * missing_index[0]
        image = image * missing_index[1]
        tokenizer = tokenizer * missing_index[2]
        padding_mask = padding_mask * missing_index[2]
        
        return tokenizer, padding_mask, image, spectrogram, label, torch.LongTensor([idx])

    
class CLIPDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.dataset = args.dataset
        if args.dataset == "Food101":
            self.data_root = '/data1/zhangxiaohui/food101/clip_feature/'
            self.visual_feature_path = os.path.join(self.data_root, "image", '{}/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text", '{}/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_food.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_food.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_food.txt"

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(".jpg ")
                token_path = os.path.join(self.text_feature_path, item[0] + '.npy')
                visual_path = os.path.join(self.visual_feature_path, item[0] + ".npy")    
                # pdb.set_trace()
                if os.path.exists(token_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    pdb.set_trace()
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))
        

    def __len__(self):
        return len(self.av_files)
    
    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '.npy')
        tokenizer = np.load(token_path)
        tokenizer = torch.tensor(tokenizer)

        # Visual
        visual_path = os.path.join(self.visual_feature_path, av_file + ".npy")
        image = np.load(visual_path)
        image = torch.tensor(image)
        # normalize the input for both training and test

        label = self.classes.index(self.data2class[av_file])
        
        return tokenizer, image, label, torch.LongTensor([idx])
   