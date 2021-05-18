import torch
import torch.nn as nn
from torchvision import models, transforms


class Resnet2(nn.Module):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, args, device, DataParallelDevice=None):
        super().__init__()
        self.device = device
        self.model = models.resnet18(pretrained=True)
        if DataParallelDevice:
            self.model = torch.nn.DataParallel(self.model, device_ids=DataParallelDevice)
        # self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # self.model = self.model.to(self.device)
        for name, param in self.model.named_parameters():
            # print(name)
            # print(param.requires_grad)
            if not name.startswith("fc"):
                param.requires_grad = False
        # import pdb; pdb.set_trace()

    def forward(self, frame, device):
        # frame = frame.to(self.device)
        feature = self.model(frame)
        # feature = feature.to(device)
        return feature


class Resnet18(object):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model = models.resnet18(pretrained=True)
        if args.gpu:
            # self.model.to(torch.device('cuda:%d' % args.gpu_id))
            # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
            # import pdb; pdb.set_trace()
            pass

        if eval:
            self.model = self.model.eval()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

    def extract(self, x):
        # import pdb; pdb.set_trace()
        return self.model(x)


class MaskRCNN(object):
    '''
    pretrained MaskRCNN from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, min_size=224):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, min_size=min_size)
        self.model = self.model.backbone.body
        self.feat_layer = 3

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()


    def extract(self, x):
        features = self.model(x)
        return features[self.feat_layer]


class Resnet(object):

    def __init__(self, args, device='cuda', eval=True, share_memory=False, use_conv_feat=True):
        # self.model_type = args.visual_model
        # self.gpu = args.gpu

        # choose model type
        self.resnet_model = Resnet18(args, eval, share_memory, use_conv_feat=use_conv_feat)
        self.resnet_model.model.to(device)
        self.device = device

        # normalization transform
        self.transform = self.get_default_transform()


    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def featurize(self, images, batch=32):
        images_normalized = torch.stack([self.transform(i) for i in images], dim=0).to(self.device)
        # if self.gpu:
        #     images_normalized = images_normalized.to(torch.device('cuda'))

        out = []
        with torch.set_grad_enabled(False):
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i:i+batch]
                out.append(self.resnet_model.extract(b))
        return torch.cat(out, dim=0)