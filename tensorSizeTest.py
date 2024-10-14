import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, mobilenet_v3_small, efficientnet_b0, resnet18, efficientnet_b3

"""
Command:

python tensorSizeTest.py

"""

choice = 6
printModel = False

#model = MyDenseNet()           --> 1
#model = MyMobileNetV3()        --> 2
#model = MyEfficientNetB0()     --> 3
#model = MyResNet()             --> 4
#model = MyEfficientNetB3Pool() --> 5
#model = RegressionNet()        --> 6


poolinglayer = 2


class MyMobileNetV3(nn.Module):
    def __init__(self, out_levels=(12,), pretrained=False):
        super(MyMobileNetV3, self).__init__()
        self.name = "mobilenet"
        # Lade ein vortrainiertes MobileNetV3-Modell
        self.model = mobilenet_v3_small(pretrained=pretrained).features[:-1]
        # Ersetze die letzte Schicht (Klassifizierungsschicht) durch eine neue Schicht mit 10 Ausgängen
        #num_ftrs = self.mobilenetv3.classifier[3].in_features
        #self.mobilenetv3.classifier[3] = nn.Linear(num_ftrs, 10)  # Annahme: 10 Klassen für die Klassifizierung
        self.stages = nn.ModuleList([self.model[i] for i in range(len(self.model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        #self.reduction_factor = 2**5 # weil in beiden anderen backbones auch enthalten
    
    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features

class MyDenseNet(nn.Module):
    def __init__(self, out_levels=(11,), pretrained=False):
        super(MyDenseNet, self).__init__()
        self.name = "densenet"
        # Lade ein vortrainiertes DenseNet-Modell
        self.model = densenet121(pretrained=pretrained).features[:-1]

        self.stages = nn.ModuleList([self.model[i] for i in range(len(self.model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        
        self.out_channels = (1024,) # überschrieben weil die letzte conv nur 32 channels hat (passt nicht)
        
        #self.reduction_factor = 2**5
    
    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features

class MyEfficientNetB0(nn.Module):
    def __init__(self, out_levels=(8,), pretrained=False):
        super(MyEfficientNetB0, self).__init__()
        self.name = "efficientnet"
        # Lade ein vortrainiertes EfficientNet-B0-Modell
        self.model = efficientnet_b0(pretrained=pretrained).features[:-1]
        # Ersetze die letzte Schicht (Klassifizierungsschicht) durch eine neue Schicht mit 10 Ausgängen
        #num_ftrs = self.efficientnet_b0.classifier.in_features
        #self.efficientnet_b0.classifier = nn.Linear(num_ftrs, 10)  # Annahme: 10 Klassen für die Klassifizierung

        self.stages = nn.ModuleList([self.model[i] for i in range(len(self.model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        #self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features

class MyResNet(nn.Module):
    def __init__(self, out_levels=(5,), pretrained=False):
        super(MyResNet, self).__init__()
        self.name = "resnet"
        # Lade ein vortrainiertes ResNet-18-Modell
        self.model = resnet18(pretrained=pretrained)
        # Ersetze die letzte Schicht (Klassifizierungsschicht) durch eine neue Schicht mit 10 Ausgängen
        #num_ftrs = self.resnet.fc.in_features
        #self.resnet.fc = nn.Linear(num_ftrs, 10)  # Annahme: 10 Klassen für die Klassifizierung

        self.stages = nn.ModuleList(
            [
                nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu),
                nn.Sequential(self.model.maxpool, self.model.layer1),
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
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
        #self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features

    #def forward(self, x):
    #    # Führe den Eingangstensor durch das ResNet-Modell
    #    x = self.resnet(x)
    #    return x

class MyEfficientNetB3Pool(nn.Module):
    def __init__(self, out_levels=(8,), pretrained=False):
        super(MyEfficientNetB3Pool, self).__init__()
        self.name = "efficientnetb3pool"
        # Lade ein vortrainiertes EfficientNet-B0-Modell
        self.model = efficientnet_b3(pretrained=pretrained).features[:-1]
        # Ersetze die letzte Schicht (Klassifizierungsschicht) durch eine neue Schicht mit 10 Ausgängen
        #num_ftrs = self.efficientnet_b0.classifier.in_features
        #self.efficientnet_b0.classifier = nn.Linear(num_ftrs, 10)  # Annahme: 10 Klassen für die Klassifizierung

        self.stages = nn.ModuleList([self.model[i] for i in range(len(self.model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        #self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features

class RegressionNet(nn.Module):
    def __init__(
        self,
        anchors,
        pool_channels,
        fc_hidden_size,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the regression method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            input_shape (tuple): Input shape (C, H, W).
            anchors (int): Number of horizontal anchors in the input image where the path is regressed.
            pool_channels (int): Number of output channels of the pooling layer.
            fc_hidden_size (int): Number of units in the hidden layer of the fully connected part.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(RegressionNet, self).__init__()
        self.backbone = MyEfficientNetB3Pool(pretrained=pretrained)
        self.name = "regressionNet"
        
        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=2048, # 8*16*16=2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=2048, # 8*16*16=2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        self.out_channels = pool_channels

    def forward(self, x):
        x = self.backbone(x)[0]
        
        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with average or max pooling layer
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1) # torch.Size([2048])
            # fea = self.pool(x) --> ohne flatten: torch.Size([2048, 1, 1])

        #reg = self.fc(fea)
        #return reg
        return fea

# Erstelle eine Instanz des Modells
if choice == 1:
    model = MyDenseNet()
elif choice == 2:
    model = MyMobileNetV3()
elif choice == 3:
    model = MyEfficientNetB0()
elif choice == 4:
    model = MyResNet()
elif choice == 5:
    model = MyEfficientNetB3Pool()
elif choice == 6:
    model = RegressionNet(64, 8, 2048)

if printModel:
    print(model)

# Definiere einen Tensor mit den Eingangsgrößen [3, 512, 512] und fülle ihn mit Zufallszahlen
input_tensor = torch.randn(8, 3, 512, 512)  # erster parameter: Batchsize 8

# Führe den Eingangstensor durch das DenseNet-Modell
output = model(input_tensor)

# print forms
print("model name: ", model.name)
print("Form des Inputtensors:", input_tensor.size())
print("größe der output liste: ", len(output))
print("out channels: ", model.out_channels)
print("Form des Ausgabetensors:", output.size())
