import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion3
from .cav_mae import CAVMAEFT
from .m3ae import MaskedMultimodalAutoencoder as M3AE
import pickle
import pdb
import einops
from ml_collections import ConfigDict
# import fairseq

class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'KineticSound':
            pass
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            pass
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            if args.gs_flag:
                self.fusion_module = ConcatFusion(input_dim = 512, output_dim=n_classes)
            else:
                self.fusion_module = ConcatFusion(input_dim = 1024, output_dim=n_classes)
        elif fusion == 'film':
            pass
        elif fusion == 'gated':
            pass
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        if args.modulation == "QMF":
            self.audio_fc = nn.Linear(512, n_classes)
            self.visual_fc = nn.Linear(512, n_classes)

        self.args = args
        

    def forward(self, audio, visual):
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        if self.args.modulation == "QMF":
            a_out = self.audio_fc(a)
            v_out = self.visual_fc(v)

            return a_out, v_out

        if not self.args.gs_flag:
            a, v, out = self.fusion_module(a, v)
            return a, v, out
        
        return a, v

class CAVClassifier(nn.Module):
    def __init__(self, args):
        super(CAVClassifier, self).__init__()
        fusion = args.fusion_method
        if args.dataset == 'KineticSound':
            pass
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            pass
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            if args.gs_flag:
                self.fusion_module = ConcatFusion(input_dim = 768, output_dim=n_classes)
            else:
                self.fusion_module = ConcatFusion(input_dim = 1536, output_dim=n_classes)
        elif fusion == 'film':
            pass
        elif fusion == 'gated':
            pass
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.mae_a = CAVMAEFT(n_classes)
        self.mae_v = CAVMAEFT(n_classes)
        
        cav_ckpt_audio = "/path/to/cavmae-audio.pth"
        cav_ckpt_visual = "/path/to/cavmae-visual.pth"
        device = torch.device('cuda:0')

        sdA_audio = torch.load(cav_ckpt_audio, map_location=device)
        miss, unexcepted = self.mae_a.load_state_dict(sdA_audio, strict=False)
        
        sdA_visual = torch.load(cav_ckpt_visual, map_location=device)
        miss, unexcepted = self.mae_v.load_state_dict(sdA_visual, strict=False)

    def forward(self, audio, visual):
        a = self.mae_a.forward_feat(audio, None, "a")
        v = self.mae_v.forward_feat(None, visual, "v")
        a = a.mean(dim = 1)
        v = v.mean(dim = 1)
        return a, v


class M3AEClassifier(nn.Module):
    def __init__(self, args):
        super(M3AEClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'MVSA':
            n_classes = 3
        elif args.dataset == 'KineticSound':
            pass
        elif args.dataset == 'Food101':
            n_classes = 101
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'CUB':
            pass
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            if args.gs_flag:
                # self.fusion_module = ConcatFusion(input_dim = 384, output_dim=n_classes)
                self.fusion_module = ConcatFusion(input_dim = 768, output_dim=n_classes)
                # self.fusion_module = ConcatFusion(input_dim = 1024, output_dim=n_classes)
            else:
                # self.fusion_module = ConcatFusion(input_dim = 768, output_dim=n_classes)
                self.fusion_module = ConcatFusion(input_dim = 1536, output_dim=n_classes)
                # self.fusion_module = ConcatFusion(input_dim = 2048, output_dim=n_classes)
        elif fusion == 'film':
            pass
        elif fusion == 'gated':
            pass
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        model_config = ConfigDict(dict(model_type='base'))
        # model_config = ConfigDict(dict(model_type='large')) 
        self.mae_a = M3AE(text_vocab_size = 30522, config_updates = model_config)
        self.mae_v = M3AE(text_vocab_size = 30522, config_updates = model_config)
        m3ae_ckpt_audio = "/path/to/m3ae_base_audio.pth"
        m3ae_ckpt_visual = "/path/to/m3ae_base_visual.pth"
        # m3ae_ckpt = "/path/to/m3ae_large.pth"
        device = torch.device('cuda:0')
        sdA_audio = torch.load(m3ae_ckpt_audio, map_location=device)
        miss, unexcepted = self.mae_a.load_state_dict(sdA_audio, strict=False)
        sdA_visual = torch.load(m3ae_ckpt_visual, map_location=device)
        miss, unexcepted = self.mae_v.load_state_dict(sdA_visual, strict=False)

        if args.modulation == "QMF":
            self.audio_fc = nn.Linear(768, n_classes)
            self.visual_fc = nn.Linear(768, n_classes)
        
        self.args = args

        
    def forward(self, token, padding_mask, visual):
        
        visual = einops.rearrange(visual, 
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1 = 16, p2 = 16)
        token = token.squeeze(1)
        padding_mask = padding_mask.squeeze(1)

        a = self.mae_a.forward_representation(None, token, padding_mask)
        v = self.mae_v.forward_representation(visual, None, None)
        
        a = a.mean(dim = 1)
        v = v.mean(dim = 1)

        if self.args.modulation == "QMF":
            a = self.audio_fc(a)
            v = self.visual_fc(v)

        return a, v

class Modal3Classifier(nn.Module):
    def __init__(self, args):
        super(Modal3Classifier, self).__init__()

        fusion = args.fusion_method

        if args.dataset == 'IEMOCAP':
            n_classes = 4
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            if args.gs_flag:
                # self.fusion_module = ConcatFusion(input_dim = 384, output_dim=n_classes)
                self.fusion_module = ConcatFusion3(input_dim = 768, output_dim=n_classes)
                # self.fusion_module = ConcatFusion(input_dim = 1024, output_dim=n_classes)
            else:
                # self.fusion_module = ConcatFusion(input_dim = 768, output_dim=n_classes)
                self.fusion_module = ConcatFusion3(input_dim = 2304, output_dim=n_classes)
                # self.fusion_module = ConcatFusion(input_dim = 2048, output_dim=n_classes)
        elif fusion == 'film':
            pass
        elif fusion == 'gated':
            pass
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        model_config = ConfigDict(dict(model_type='base'))
        self.mae_a = CAVMAEFT(n_classes)
        self.mae_v = M3AE(text_vocab_size = 30522, config_updates = model_config)
        self.mae_t = M3AE(text_vocab_size = 30522, config_updates = model_config)
        cav_ckpt_audio = "/path/to/cavmae-audio.pth"
        m3ae_ckpt = "/path/to/m3ae_base.pth"
        device = torch.device('cuda:0')
        sdA_audio = torch.load(cav_ckpt_audio, map_location=device)
        miss, unexcepted = self.mae_a.load_state_dict(sdA_audio, strict=False)
        sdA_visual = torch.load(m3ae_ckpt, map_location=device)
        miss, unexcepted = self.mae_v.load_state_dict(sdA_visual, strict=False)
        sdA_text = torch.load(m3ae_ckpt, map_location=device)
        miss, unexcepted = self.mae_t.load_state_dict(sdA_text, strict=False)

        if args.modulation == "QMF":
            self.audio_fc = nn.Linear(768, n_classes)
            self.visual_fc = nn.Linear(768, n_classes)
            self.txtual_fc = nn.Linear(768, n_classes)
        
        self.args = args

        
    def forward(self, token, padding_mask, visual, audio):
        
        visual = einops.rearrange(visual, 
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1 = 16, p2 = 16)

        token = token.squeeze(1)
        padding_mask = padding_mask.squeeze(1)

        a = self.mae_a.forward_feat(audio, None, "a")
        t = self.mae_t.forward_representation(None, token, padding_mask)
        v = self.mae_v.forward_representation(visual, None, None)
        
        a = a.mean(dim = 1)
        v = v.mean(dim = 1)
        t = t.mean(dim = 1)

        if self.args.modulation == "QMF":
            a = self.audio_fc(a)
            v = self.visual_fc(v)
            t = self.txtual_fc(t)
            return a, v, t

        return a, v, t


class CLIPClassifier(nn.Module):
    def __init__(self, args):
        super(CLIPClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'MVSA':
            n_classes = 3
        elif args.dataset == 'KineticSound':
            pass
        elif args.dataset == 'Food101':
            n_classes = 101
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'CUB':
            pass
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            if args.gs_flag:
                self.fusion_module = ConcatFusion(input_dim = 512, output_dim=n_classes)
            else:
                self.fusion_module = ConcatFusion(input_dim = 1024, output_dim=n_classes)
        elif fusion == 'film':
            pass
        elif fusion == 'gated':
            pass
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        
        self.args = args

        
    def forward(self, token, visual):
        token = token.squeeze(1)
        visual = visual.squeeze(1)
        if not self.args.gs_flag:
            a, v, out = self.fusion_module(token, visual)
            return a, v, out
        return token, visual



