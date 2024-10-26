import math
import torch
import torch.nn as nn
import time

from .backbone import EfficientNetBackbone, ResNetBackbone, MobileNetV3_Backbone, Densenet_Backbone

poolinglayer = 2 # 0 -> TEP original, kein Pooling Layer
                 # 1 -> Adaptive Average - Pooling Layer
                 # 2 -> Adaptive Max     - Pooling layer

headLinearlayers = 3 # 0 -> TEP original, 2 linear layers
                     # 1 -> depth-head (mehrere Layer mit der gleichen Größe (2048))
                     # 2 -> width-head (nur 2 oder 3 Linear Layers, aber mit einer größeren Feature size)
                     # 3 -> trapez-head (wird zuerst groß und dann immer kleiner)

# CNN_LSTM_FC:
CNN_LSTM_FC_num_lstm_layers = 2
#CNN_LSTM_FC_lstm_hidden_size = um die 50 (von 50 bis 100)

# CNN_FC_LSTM:
CNN_FC_LSTM_num_lstm_layers = 2
CNN_FC_LSTM_lstm_hidden_size = 65 # Hälfte von 129 # um die 50 (von 50 bis 100)

# CNN_LSTM:
CNN_LSTM_num_lstm_layers = 2
CNN_LSTM_lstm_input_size = 65
CNN_LSTM_lstm_hidden_size = 65

fc_out = True

sliding_window_size = 10

"""
#. Linear Layer: input, output

DEPTH HEAD
1. Linear Layer: 2048, 2048
2. Linear Layer: 2048, 2048
3. Linear Layer: 2048, 2048
4. Linear Layer: 2048, 129 (anchors * 2 + 1) [output]

WIDTH HEAD
1. Linear Layer: 2048, 4096 (2048 * 2)
2. Linear Layer: 4096, 4096 (2048 * 2)
3. Linear Layer: 4096, 129 (anchors * 2 + 1) [output]

TRAPEZ HEAD
1. Linear Layer: 2048, 3584 (2048*1,75)
2. Linear Layer: 3584, 2560 (2048*1,25)
3. Linear Layer: 2560, 2048
4. Linear Layer: 2048, 129 (anchors * 2 + 1) [output]
"""

# V0 model
class RegressionNetCNN_LSTM_FC(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_LSTM_FC, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        lstm_hidden_size = self.num_channels # 2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.num_channels, # 2048
                            hidden_size=lstm_hidden_size, # 2048
                            num_layers=CNN_LSTM_FC_num_lstm_layers, batch_first=True)

        # Define fully connected layers
        if headLinearlayers == 0:               #original TEP
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden_size, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, anchors * 2 + 1), # 2. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 1:               #depth-head
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden_size, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 2. Linear Layer (2048, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 3. Linear Layer (2048, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, anchors * 2 + 1), # 4. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 2:               #width-head
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden_size, fc_hidden_size * 2), # 1. Linear Layer (2048, 2048*2)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, fc_hidden_size * 2), # 2. Linear Layer (2048*2, 2048*2)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, anchors * 2 + 1), # 3. Linear Layer (2048*2, 129 [output])
            )
        elif headLinearlayers == 3:               #trapez-head
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden_size, int(fc_hidden_size * 1.75)), # 1. Linear Layer (2048, 2048*1.75=3584)
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.75), int(fc_hidden_size * 1.25)), # 2. Linear Layer (2048*1.75, 2048*1.25)
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.25), fc_hidden_size), # 3. Linear Layer (2048*1.25, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, anchors * 2 + 1), # 4. Linear Layer (2048, 129 [output])
            )

    def forward(self, x):
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")

        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        shape_cnn_output = fea.shape

        # Reshape for LSTM (batch_size, seq_len, input_size=self.num_channels)
        if len(shape) == 5:
            # training:
            # batch_size = übernehmen, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(shape[0], shape[1], shape_cnn_output[1])
            
        elif len(shape) == 4:
            # inference:
            # batch_size = 1, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(1, shape[0], shape_cnn_output[1])

        # Apply LSTM
        lstm_out, _ = self.lstm(fea)

        # Use the last output of the LSTM
        # [alle batches, nur letzte (aktuellste) sequenz, alle channels]
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layers
        reg = self.fc(lstm_out)
        return reg

# V1 model
class RegressionNetCNN_FC_LSTM(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_FC_LSTM, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        cnn_output_size = 65 # pretrained anchors * 2 + 1 # kind-donkey-84: 129; decent-bee-298: 65; toasty-haze-299: 43

        # Define fully connected layers
        if headLinearlayers == 0:               #original TEP
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 2. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 1:               #depth-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 2. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 3. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 4. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 2:               #width-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size * 2), # 1. Linear Layer (2048, 2048*2)
                #nn.BatchNorm1d(fc_hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, fc_hidden_size * 2), # 2. Linear Layer (2048*2, 2048*2)
                #nn.BatchNorm1d(fc_hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, cnn_output_size), # 3. Linear Layer (2048*2, 129 [output])
            )
        elif headLinearlayers == 3:               #trapez-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, int(fc_hidden_size * 1.75)), # 1. Linear Layer (2048, 2048*1.75=3584)
                #nn.BatchNorm1d(int(fc_hidden_size * 1.75)),
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.75), int(fc_hidden_size * 1.25)), # 2. Linear Layer (2048*1.75, 2048*1.25)
                #nn.BatchNorm1d(int(fc_hidden_size * 1.25)),
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.25), fc_hidden_size), # 3. Linear Layer (2048*1.25, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 4. Linear Layer (2048, 129 [output])
            )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=cnn_output_size,  # 129
                            hidden_size=CNN_FC_LSTM_lstm_hidden_size, # 1. 129, 2. 65, 3. 65, 4.43
                            num_layers=CNN_FC_LSTM_num_lstm_layers, batch_first=True)
        
        self.fc_out = nn.Sequential(
            nn.ReLU(inplace=False), # inplace = False sonst Probleme mit LSTM
            nn.Linear(CNN_FC_LSTM_lstm_hidden_size, anchors * 2 + 1), # 5. Linear Layer after LSTM (# 1. 129, 2. 129, 3. 65, 4. 43 [output])
        )

        #self.training = True
        #self.saved_features_cnn_training = []   # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_training = []    # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]
        #self.saved_features_cnn_validation = [] # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_validation = []  # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]

    def forward(self, x):
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        #print("after cnn output dimensions: ", fea.shape)
        
        # Speichere die Feature Maps nach dem CNN und Pooling: 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #if self.training:
        #    self.saved_features_cnn_training.append(fea.clone())  # clone() um sicherzustellen, dass die Tensoren nicht verändert werden
        #else:
        #    self.saved_features_cnn_validation.append(fea.clone())  # clone() um sicherzustellen, dass die Tensoren nicht verändert werden

        # Fully connected layers
        fea = self.fc(fea)

        #print("after fc output dimensions: ", fea.shape)

        # Speichere die Outputs nach den FC-Layers ab: 2D Tensoren mit [batch_size * seq_len (1*10), output_size (129)]
        #if self.training:
        #    self.saved_featrues_fc_training.append(fea.clone())  # clone() um sicherzustellen, dass die Tensoren nicht verändert werden
        #else:
        #    self.saved_featrues_fc_validation.append(fea.clone())  # clone() um sicherzustellen, dass die Tensoren nicht verändert werden

        shape_fc_output = fea.shape # [batch_size * seq_len, output_size (129)]

        # Reshape for LSTM (batch_size, seq_len, input_size=self.num_channels)
        if len(shape) == 5:
            # training:
            # batch_size = übernehmen, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(shape[0], shape[1], shape_fc_output[1])
            
        elif len(shape) == 4:
            # inference:
            # batch_size = 1, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(1, shape[0], shape_fc_output[1])

        # Apply LSTM
        lstm_out, _ = self.lstm(fea)

        #print("after lstm output dimensions: ", lstm_out.shape)

        # Use the last output of the LSTM
        # [alle batches, nur letzte (aktuellste) sequenz, alle channels]
        lstm_out = lstm_out[:, -1, :]

        # Apply FC_OUT
        reg = self.fc_out(lstm_out)

        #print("after fc_out output dimensions: ", reg.shape)
        #print("---")
        
        return reg

# V1 model
class RegressionNetCNN_FC_FCOUT(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_FC_FCOUT, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        cnn_output_size = 129 # pretrained anchors * 2 + 1 # kind-donkey-84: 129; decent-bee-298: 65; toasty-haze-299: 43

        # Define fully connected layers
        if headLinearlayers == 0:               #original TEP
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 2. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 1:               #depth-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 2. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 3. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 4. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 2:               #width-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size * 2), # 1. Linear Layer (2048, 2048*2)
                #nn.BatchNorm1d(fc_hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, fc_hidden_size * 2), # 2. Linear Layer (2048*2, 2048*2)
                #nn.BatchNorm1d(fc_hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, cnn_output_size), # 3. Linear Layer (2048*2, 129 [output])
            )
        elif headLinearlayers == 3:               #trapez-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, int(fc_hidden_size * 1.75)), # 1. Linear Layer (2048, 2048*1.75=3584)
                #nn.BatchNorm1d(int(fc_hidden_size * 1.75)),
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.75), int(fc_hidden_size * 1.25)), # 2. Linear Layer (2048*1.75, 2048*1.25)
                #nn.BatchNorm1d(int(fc_hidden_size * 1.25)),
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.25), fc_hidden_size), # 3. Linear Layer (2048*1.25, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 4. Linear Layer (2048, 129 [output])
            )
        
        self.fc_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(cnn_output_size*sliding_window_size, anchors * 2 + 1), # 5. Linear Layer for learning temporal data -> cnn_output_size*10, weil used_images: 10
        )

        #self.training = True
        #self.saved_features_cnn_training = []   # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_training = []    # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]
        #self.saved_features_cnn_validation = [] # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_validation = []  # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]

    def forward(self, x):
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        #print("after cnn output dimensions: ", fea.shape)
        
        # Fully connected layers
        fea = self.fc(fea)

        #print("after fc output dimensions: ", fea.shape)

        fea = fea.view(fea.shape[0] * fea.shape[1])

        #print("tensor after view: ", fea.shape)

        fea = fea.unsqueeze(0)

        #print("tensor after unsqueeze: ", fea.shape)

        # Apply FC_OUT
        reg = self.fc_out(fea)

        #print("after fc_out output dimensions: ", reg.shape)
        #print("---")
        
        return reg

# V1 model
class RegressionNetCNN_LSTM(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_LSTM, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.num_channels, CNN_LSTM_lstm_input_size), # 1. Linear Layer (2048, 65)
            nn.ReLU(inplace=True),
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=CNN_LSTM_lstm_input_size,  # input size = 65
                            hidden_size=CNN_LSTM_lstm_hidden_size, # hidden size = 65
                            num_layers=CNN_LSTM_num_lstm_layers, batch_first=True) # layers = 2
        
        if fc_out:
            self.fc_out = nn.Sequential(
                nn.ReLU(inplace=False), # inplace = False sonst Probleme mit LSTM
                nn.Linear(CNN_LSTM_lstm_hidden_size, anchors * 2 + 1), # 2. Linear Layer after LSTM (65, 129 [output])
            )

    def forward(self, x):
        #start_time_backbone = time.time()
        
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        #print("after cnn output dimensions: ", fea.shape)

        # Fully connected layers
        fea = self.fc(fea)

        #print("after fc output dimensions: ", fea.shape)

        shape_fc_output = fea.shape # [batch_size * seq_len, output_size (129)]

        # Reshape for LSTM (batch_size, seq_len, input_size=self.num_channels)
        if len(shape) == 5:
            # training:
            # batch_size = übernehmen, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(shape[0], shape[1], shape_fc_output[1])
            
        elif len(shape) == 4:
            # inference:
            # batch_size = 1, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(1, shape[0], shape_fc_output[1])

        # Time before LSTM
        #start_time_lstm = time.time()

        # Apply LSTM
        lstm_out, _ = self.lstm(fea)

        # Time after LSTM
        #end_time_lstm = time.time()

        #print("after lstm output dimensions: ", lstm_out.shape)

        # Use the last output of the LSTM
        # [alle batches, nur letzte (aktuellste) sequenz, alle channels]
        lstm_out = lstm_out[:, -1, :]

        # Apply FC_OUT
        if fc_out:
            reg = self.fc_out(lstm_out)
        else:
            reg = lstm_out

        #end_time_fcout = time.time()
        
        """
        # Measure the duration of LSTM processing
        backbone_duration = start_time_lstm - start_time_backbone
        print(f"Backbone processing time: {backbone_duration:.6f} seconds")
        lstm_duration = end_time_lstm - start_time_lstm
        print(f"LSTM processing time: {lstm_duration:.6f} seconds")
        fcout_duration = end_time_fcout - end_time_lstm
        print(f"fcout processing time: {fcout_duration:.6f} seconds")
        overall_duration = end_time_fcout - start_time_backbone
        print(f"overall processing time: {overall_duration:.6f} seconds")
        print("---")
        """
        #print("after fc_out output dimensions: ", reg.shape)
        #print("---")
        
        return reg

# V2 model
class RegressionNetCNN_LSTM_V2(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_LSTM_V2, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.num_channels, int(self.num_channels/2)), # 1. Linear Layer (2048, 2048/2 = 1024)
            nn.ReLU(inplace=True),
            nn.Linear(int(self.num_channels/2), int(self.num_channels/4)), # 1. Linear Layer (2048/2 = 1024, 2048/4 = 512)
            nn.ReLU(inplace=True),
            nn.Linear(int(self.num_channels/4), CNN_LSTM_lstm_input_size), # 1. Linear Layer (512, 65)
            nn.ReLU(inplace=True),
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=CNN_LSTM_lstm_input_size,  # input size = 65
                            hidden_size=CNN_LSTM_lstm_hidden_size, # hidden size = 65
                            num_layers=CNN_LSTM_num_lstm_layers, batch_first=True) # layers = 2
        
        self.scale = nn.Parameter(1/2*torch.ones(CNN_LSTM_lstm_input_size),requires_grad=True)
        self.offset = nn.Parameter(0.5*torch.ones(CNN_LSTM_lstm_input_size),requires_grad=True)

        if fc_out:
            self.fc_out = nn.Sequential(
                nn.ReLU(inplace=False), # inplace = False sonst Probleme mit LSTM
                nn.Linear(CNN_LSTM_lstm_hidden_size, anchors * 2 + 1), # 2. Linear Layer after LSTM (65, 129 [output])
            )

    def forward(self, x):
        #start_time_backbone = time.time()
        
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        #print("after cnn output dimensions: ", fea.shape)

        # Fully connected layers
        fea = self.fc(fea)

        #print("after fc output dimensions: ", fea.shape)

        shape_fc_output = fea.shape # [batch_size * seq_len, output_size (129)]

        # Reshape for LSTM (batch_size, seq_len, input_size=self.num_channels)
        if len(shape) == 5:
            # training:
            # batch_size = übernehmen, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(shape[0], shape[1], shape_fc_output[1])
            
        elif len(shape) == 4:
            # inference:
            # batch_size = 1, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(1, shape[0], shape_fc_output[1])

        # Time before LSTM
        #start_time_lstm = time.time()

        # Apply LSTM
        lstm_out, _ = self.lstm(fea)

        lstm_out = lstm_out*self.scale.view(1,1,-1) + self.offset.view(1,1,-1)

        # Time after LSTM
        #end_time_lstm = time.time()

        #print("after lstm output dimensions: ", lstm_out.shape)

        # Use the last output of the LSTM
        # [alle batches, nur letzte (aktuellste) sequenz, alle channels]
        lstm_out = lstm_out[:, -1, :]

        # Apply FC_OUT
        if fc_out:
            reg = self.fc_out(lstm_out)
        else:
            reg = lstm_out

        #end_time_fcout = time.time()
        
        """
        # Measure the duration of LSTM processing
        backbone_duration = start_time_lstm - start_time_backbone
        print(f"Backbone processing time: {backbone_duration:.6f} seconds")
        lstm_duration = end_time_lstm - start_time_lstm
        print(f"LSTM processing time: {lstm_duration:.6f} seconds")
        fcout_duration = end_time_fcout - end_time_lstm
        print(f"fcout processing time: {fcout_duration:.6f} seconds")
        overall_duration = end_time_fcout - start_time_backbone
        print(f"overall processing time: {overall_duration:.6f} seconds")
        print("---")
        """
        #print("after fc_out output dimensions: ", reg.shape)
        #print("---")
        
        return reg

# V2 model
class RegressionNetCNN_LSTM_HEAD_V2(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_LSTM_HEAD_V2, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.num_channels, int(self.num_channels/2)), # 1. Linear Layer (2048, 1024)
            nn.ReLU(inplace=True),
            nn.Linear(int(self.num_channels/2), CNN_LSTM_lstm_input_size), # 1. Linear Layer (1024, 65)
            nn.ReLU(inplace=True),
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=CNN_LSTM_lstm_input_size,  # input size = 65
                            hidden_size=CNN_LSTM_lstm_hidden_size, # hidden size = 65
                            num_layers=CNN_LSTM_num_lstm_layers, batch_first=True) # layers = 2
        
        self.scale = nn.Parameter(1/2*torch.ones(CNN_LSTM_lstm_input_size),requires_grad=True)
        self.offset = nn.Parameter(0.5*torch.ones(CNN_LSTM_lstm_input_size),requires_grad=True)

        if fc_out:
            self.fc_out = nn.Sequential(
                nn.ReLU(inplace=False), # inplace = False sonst Probleme mit LSTM
                nn.Linear(CNN_LSTM_lstm_hidden_size, anchors * 2 + 1), # 2. Linear Layer after LSTM (65, 129 [output])
                nn.ReLU(inplace=True),
                nn.Linear(anchors * 2 + 1, anchors * 2 + 1), # 2. Linear Layer after LSTM (129, 129 [output])
            )

    def forward(self, x):
        #start_time_backbone = time.time()
        
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        #print("after cnn output dimensions: ", fea.shape)

        # Fully connected layers
        fea = self.fc(fea)

        #print("after fc output dimensions: ", fea.shape)

        shape_fc_output = fea.shape # [batch_size * seq_len, output_size (129)]

        # Reshape for LSTM (batch_size, seq_len, input_size=self.num_channels)
        if len(shape) == 5:
            # training:
            # batch_size = übernehmen, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(shape[0], shape[1], shape_fc_output[1])
            
        elif len(shape) == 4:
            # inference:
            # batch_size = 1, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(1, shape[0], shape_fc_output[1])

        # Time before LSTM
        #start_time_lstm = time.time()

        # Apply LSTM
        lstm_out, _ = self.lstm(fea)

        lstm_out = lstm_out*self.scale.view(1,1,-1) + self.offset.view(1,1,-1)

        # Time after LSTM
        #end_time_lstm = time.time()

        #print("after lstm output dimensions: ", lstm_out.shape)

        # Use the last output of the LSTM
        # [alle batches, nur letzte (aktuellste) sequenz, alle channels]
        lstm_out = lstm_out[:, -1, :]

        # Apply FC_OUT
        if fc_out:
            reg = self.fc_out(lstm_out)
        else:
            reg = lstm_out

        #end_time_fcout = time.time()
        
        """
        # Measure the duration of LSTM processing
        backbone_duration = start_time_lstm - start_time_backbone
        print(f"Backbone processing time: {backbone_duration:.6f} seconds")
        lstm_duration = end_time_lstm - start_time_lstm
        print(f"LSTM processing time: {lstm_duration:.6f} seconds")
        fcout_duration = end_time_fcout - end_time_lstm
        print(f"fcout processing time: {fcout_duration:.6f} seconds")
        overall_duration = end_time_fcout - start_time_backbone
        print(f"overall processing time: {overall_duration:.6f} seconds")
        print("---")
        """
        #print("after fc_out output dimensions: ", reg.shape)
        #print("---")
        
        return reg

# V2.2 model
class RegressionNetCNN_LSTM_HEAD(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_LSTM, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.num_channels, (self.num_channels/2)), # 1. Linear Layer (2048, 1024)
            nn.ReLU(inplace=True),
            nn.Linear((self.num_channels/2), CNN_LSTM_lstm_input_size), # 1. Linear Layer (2048, 65)
            nn.ReLU(inplace=True),
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=CNN_LSTM_lstm_input_size,  # input size = 65
                            hidden_size=CNN_LSTM_lstm_hidden_size, # hidden size = 65
                            num_layers=CNN_LSTM_num_lstm_layers, batch_first=True) # layers = 2
        
        if fc_out:
            self.fc_out = nn.Sequential(
                nn.ReLU(inplace=False), # inplace = False sonst Probleme mit LSTM
                nn.Linear(CNN_LSTM_lstm_hidden_size, anchors * 2 + 1), # 2. Linear Layer after LSTM (65, 129 [output])
                nn.ReLU(inplace=True),
                nn.Linear(anchors * 2 + 1, anchors * 2 + 1), # 2. Linear Layer after LSTM (129, 129 [output])
            )

    def forward(self, x):
        #start_time_backbone = time.time()
        
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        #print("after cnn output dimensions: ", fea.shape)

        # Fully connected layers
        fea = self.fc(fea)

        #print("after fc output dimensions: ", fea.shape)

        shape_fc_output = fea.shape # [batch_size * seq_len, output_size (129)]

        # Reshape for LSTM (batch_size, seq_len, input_size=self.num_channels)
        if len(shape) == 5:
            # training:
            # batch_size = übernehmen, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(shape[0], shape[1], shape_fc_output[1])
            
        elif len(shape) == 4:
            # inference:
            # batch_size = 1, seq_len = übernehmen, num_channels = übernehmen von CNN
            fea = fea.view(1, shape[0], shape_fc_output[1])

        # Time before LSTM
        #start_time_lstm = time.time()

        # Apply LSTM
        lstm_out, _ = self.lstm(fea)

        # Time after LSTM
        #end_time_lstm = time.time()

        #print("after lstm output dimensions: ", lstm_out.shape)

        # Use the last output of the LSTM
        # [alle batches, nur letzte (aktuellste) sequenz, alle channels]
        lstm_out = lstm_out[:, -1, :]

        # Apply FC_OUT
        if fc_out:
            reg = self.fc_out(lstm_out)
        else:
            reg = lstm_out

        #end_time_fcout = time.time()
        
        """
        # Measure the duration of LSTM processing
        backbone_duration = start_time_lstm - start_time_backbone
        print(f"Backbone processing time: {backbone_duration:.6f} seconds")
        lstm_duration = end_time_lstm - start_time_lstm
        print(f"LSTM processing time: {lstm_duration:.6f} seconds")
        fcout_duration = end_time_fcout - end_time_lstm
        print(f"fcout processing time: {fcout_duration:.6f} seconds")
        overall_duration = end_time_fcout - start_time_backbone
        print(f"overall processing time: {overall_duration:.6f} seconds")
        print("---")
        """
        #print("after fc_out output dimensions: ", reg.shape)
        #print("---")
        
        return reg
    
# V2 model
class RegressionNetCNN_FC_FCOUT_V2(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_FC_FCOUT, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        cnn_output_size = 129 # pretrained anchors * 2 + 1 # kind-donkey-84: 129; decent-bee-298: 65; toasty-haze-299: 43

        # Define fully connected layers
        if headLinearlayers == 0:               #original TEP
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 2. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 1:               #depth-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size), # 1. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 2. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, fc_hidden_size), # 3. Linear Layer (2048, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 4. Linear Layer (2048, 129 [output])
            )
        elif headLinearlayers == 2:               #width-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, fc_hidden_size * 2), # 1. Linear Layer (2048, 2048*2)
                #nn.BatchNorm1d(fc_hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, fc_hidden_size * 2), # 2. Linear Layer (2048*2, 2048*2)
                #nn.BatchNorm1d(fc_hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size * 2, cnn_output_size), # 3. Linear Layer (2048*2, 129 [output])
            )
        elif headLinearlayers == 3:               #trapez-head
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels, int(fc_hidden_size * 1.75)), # 1. Linear Layer (2048, 2048*1.75=3584)
                #nn.BatchNorm1d(int(fc_hidden_size * 1.75)),
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.75), int(fc_hidden_size * 1.25)), # 2. Linear Layer (2048*1.75, 2048*1.25)
                #nn.BatchNorm1d(int(fc_hidden_size * 1.25)),
                nn.ReLU(inplace=True),
                nn.Linear(int(fc_hidden_size * 1.25), fc_hidden_size), # 3. Linear Layer (2048*1.25, 2048)
                #nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden_size, cnn_output_size), # 4. Linear Layer (2048, 129 [output])
            )
        
        self.fc_out = nn.Sequential(
            nn.Linear(cnn_output_size*sliding_window_size, cnn_output_size*sliding_window_size), # 5. Linear Layer for learning temporal data -> cnn_output_size*10, weil used_images: 10
            #nn.BatchNorm1d(fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cnn_output_size*sliding_window_size, cnn_output_size*sliding_window_size/2), # 5. Linear Layer for learning temporal data -> cnn_output_size*10, weil used_images: 10
            #nn.BatchNorm1d(fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cnn_output_size*sliding_window_size/2, anchors * 2 + 1), # 5. Linear Layer for learning temporal data -> cnn_output_size*10, weil used_images: 10
        )

        #self.training = True
        #self.saved_features_cnn_training = []   # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_training = []    # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]
        #self.saved_features_cnn_validation = [] # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_validation = []  # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]

    def forward(self, x):
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)
        
        #print("after cnn output dimensions: ", fea.shape)
        
        # Fully connected layers
        fea = self.fc(fea)

        #print("after fc output dimensions: ", fea.shape)

        fea = fea.view(fea.shape[0] * fea.shape[1])

        #print("tensor after view: ", fea.shape)

        fea = fea.unsqueeze(0)

        #print("tensor after unsqueeze: ", fea.shape)

        # Apply FC_OUT
        reg = self.fc_out(fea)

        #print("after fc_out output dimensions: ", reg.shape)
        #print("---")
        
        return reg

# V3 model
class RegressionNetCNN_FLAT_FC(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
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
        super(RegressionNetCNN_FC_FCOUT, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(version=backbone[13:], pretrained=pretrained)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        elif backbone.startswith("mobilenet"):
            self.backbone = MobileNetV3_Backbone(version=backbone[10:], pretrained=pretrained)
        elif backbone.startswith("densenet"):
            self.backbone = Densenet_Backbone(version=backbone[8:], pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.num_channels = pool_channels * math.ceil(input_shape[1] / self.backbone.reduction_factor) * math.ceil(input_shape[2] / self.backbone.reduction_factor) # 8*(512/32)*(512/32)=2048 // 8*16*16=2048

        if poolinglayer == 0:               #original TEP
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif poolinglayer == 1:             # mit pooling layer
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global adaptive average pooling
        elif poolinglayer == 2:
            self.conv = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=self.num_channels, # 2048
                kernel_size=1,
            )  # stride=1, padding=0
            self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Global adaptive max pooling

        cnn_output_size_after_flat = self.num_channels*sliding_window_size # num_images: 10

        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size_after_flat, int(cnn_output_size_after_flat/4)), # 1. Linear Layer (20480, 20480/4=5120)
            nn.ReLU(inplace=True),
            nn.Linear(int(cnn_output_size_after_flat/4), int(cnn_output_size_after_flat/8)), # 2. Linear Layer (5120, 20480/8=2560)
            nn.ReLU(inplace=True),
            nn.Linear(int(cnn_output_size_after_flat/8), int(cnn_output_size_after_flat/16)), # 3. Linear Layer (2560, 2048)
            nn.ReLU(inplace=True),
            nn.Linear(int(cnn_output_size_after_flat/16), anchors * 2 + 1), # 4. Linear Layer (2048, 129 [output])
        )

        #self.training = True
        #self.saved_features_cnn_training = []   # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_training = []    # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]
        #self.saved_features_cnn_validation = [] # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
        #self.saved_featrues_fc_validation = []  # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), cnn_output_size (129)]

    def forward(self, x):
        # x must be [batch_size], seq_len, inputsize (-1) bzw. channel_size
        shape = x.shape

        if len(x.shape) == 5:
            x = x.view(shape[0]*shape[1], *shape[2:]) # (batch_size * seq_len, C, H, W)
        else:
            ValueError("Input Tensor for Backbone has uncorrect dimensions !!!")
        self.backbone.eval() # um auch batch_norm zu freezen
        x = self.backbone(x)[0]

        if poolinglayer == 0:
            # original TEP
            fea = self.pool(x).flatten(start_dim=1)
        elif poolinglayer == 1 or poolinglayer == 2:
            # with pooling layers
            x = self.conv(x)
            fea = self.pool(x).flatten(start_dim=1)

        #print("after cnn output dimensions: ", fea.shape)

        # Flatten to one big vector (every frame in one big frame)
        fea = fea.view(fea.shape[0] * fea.shape[1])
        #print("tensor after view: ", fea.shape)
        fea = fea.unsqueeze(0)
        #print("tensor after unsqueeze: ", fea.shape)
        
        # Fully connected layers
        reg = self.fc(fea)
        
        return reg

# model with skip connection
# model with GRU - am ende bestes model mit GRU auch probieren

# model with 3d Conv - nicht mehr masterarbeit