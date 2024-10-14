import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, version, out_levels=(5,), pretrained=False):
        """Initializes the ResNet backbone.

        Args:
            version (str): Version of the ResNet backbone.
            out_levels (tuple): Which stage outputs to return. Defaults to (5,) (i.e. the last stage).
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(ResNetBackbone, self).__init__()
        model_versions = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        model = model_fn(weights=weights if pretrained else None)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(model.conv1, model.bn1, model.relu),
                nn.Sequential(model.maxpool, model.layer1),
                model.layer2,
                model.layer3,
                model.layer4,
            ]
        )
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        #print("resnet_out_channels: ", self.out_channels)
        self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features


class EfficientNetBackbone(nn.Module):
    def __init__(self, version, out_levels=(8,), pretrained=False):
        """Initializes the EfficientNet backbone.

        Args:
            version (str): Version of the EfficientNet backbone.
            out_levels (tuple): Which stage outputs to return. Defaults to (8,) (i.e. the last stage).
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(EfficientNetBackbone, self).__init__()
        model_versions = {
            "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        # last block is discarded because it would be redundant with the pooling layer
        model = model_fn(weights=weights if pretrained else None).features[:-1]
        self.stages = nn.ModuleList([model[i] for i in range(len(model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features

class MobileNetV3_Backbone(nn.Module):
    def __init__(self, version, pretrained=False):
        """Initializes the MobileNetV3 backbone.

        Args:
            version (str): Version of the MobileNetV3 backbone.
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(MobileNetV3_Backbone, self).__init__()
        model_versions = {
            "small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
            "large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        }

        if version == "small":
            out_levels = (12,)
            print("MobileNetV3-small is in use")
        elif version == "large":
            out_levels = (16,)
            print("MobileNetV3-large is in use")
        else:
            print("Error in out_levels")
            raise NotImplementedError

        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]

        # last block is discarded because it would be redundant with the pooling layer (see prediction-head in model.py)
        # additionally only features are kept, this removes last four layers for small and large model (small/large):
        # input: 7² x 96   / 7² x 160;  conv2d, 1x1, out_channels: 576 / 960
        # input: 7² x 576  / 7² x 960;  pool,   7x7, out_channels: 576 / 960
        # input: 1² x 576  / 1² x 960;  conv2d, 1x1, out_channels: 1024 / 1280
        # input: 1² x 1024 / 1² x 1280; conv2d, 1x1, out_channels: k
        model = model_fn(weights=weights if pretrained else None).features[:-1]

        # taking layers like in other backbones
        self.stages = nn.ModuleList([model[i] for i in range(len(model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 2**5 # weil in beiden anderen backbones auch enthalten
    
    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features

class Densenet_Backbone(nn.Module):
    def __init__(self, version, out_levels=(11,), pretrained=False):
    #def __init__(self, version, out_levels=(11,), pretrained=False):
        """Initializes the DenseNet backbone.

        Args:
            version (str): Version of the DenseNet backbone.
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(Densenet_Backbone, self).__init__()
        model_versions = {
            "121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            "161": (models.densenet161, models.DenseNet161_Weights.DEFAULT),
            "169": (models.densenet169, models.DenseNet169_Weights.DEFAULT),
            "201": (models.densenet201, models.DenseNet201_Weights.DEFAULT),
        }

        if version == "121":
            print("densenet121 is in use")
        elif version == "161":
            print("densenet161 is in use")
        elif version == "169":
            print("densenet169 is in use")
        elif version == "201":
            print("densenet201 is in use")
        else:
            raise NotImplementedError

        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]

        # classification layer at the end is discarded
        # (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # input: 1² x 1024; conv2d, 1x1, out_channels: k --> classification layer
        model = model_fn(weights=weights if pretrained else None).features[:-1] # second last layer [BatchNorm2d] is also discarded

        #print(model)

        # taking layers like in other backbones
        self.stages = nn.ModuleList([model[i] for i in range(len(model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)

        if version == "121":
            self.out_channels = (1024,) # überschrieben weil die letzte conv nur 32 channels hat (passt nicht)
        elif version == "161":
            self.out_channels = (2208,) # überschrieben weil die letzte conv nur 32 channels hat (passt nicht)
        elif version == "169":
            self.out_channels = (1664,) # überschrieben weil die letzte conv nur 32 channels hat (passt nicht)
        elif version == "201":
            self.out_channels = (1920,) # überschrieben weil die letzte conv nur 32 channels hat (passt nicht)
        else:
            raise NotImplementedError

        print("out_channels: ", self.out_channels)
        
        self.reduction_factor = 2**5
    
    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features