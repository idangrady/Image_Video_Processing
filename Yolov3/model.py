import torch
import torch.nn as nn

# tuple (out_cha, Kernel_size, stride)
# list  [block, num_of _repeat]
# string scale prediction / upsampling

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    """
    Convolution Layers
    """

    def __init__(self, in_channels, out_channels, batch_activate=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=not batch_activate, **kwargs)
        self.bt_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(.1)
        self.batch_activate = batch_activate

    def forward(self, X):

        """
        In the forward pass we should consider both scenarios, with batch norm or without

        The reason for that is that in the prediction stage, we do not want to use batch normalisation yet still use the CNNBlock
        """
        if(self.batch_activate):
            return self.relu(self.bt_norm(self.conv(X)))
        else:
            return self.conv(X)
class ScalePrediction(nn.Module):
    """
    for scale prediction -> branch
    """
    def __init__(self, in_channels, num_channels):
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels

        self.pred = nn.Sequential(
            CNNBlock(in_channels=in_channels,
                     out_channels= in_channels*2,
                     kernel_size =3,
                     padding = 1),

            CNNBlock(in_channels=in_channels * 2,
                     out_channels=3 * (num_channels+5 ), # we add 5 as the prediction has x,y,w,h,probability
                     kernel_size=1,
                     batch_activate = False
                     ),
        )
    def forward(self, x):
        """
        we want to reshape
        (num_of_batches, 3 , self_classes+5,
        """
        out = self.pred(x).reshape(
                            x.shape[0],
                            3, self.num_channels+5,
                            x.shape[2],
                            x.shape[3]
                                   )
        return out.permute(0,1,3,4,2)

class ResidualBlock(nn.Module):
    """
    for B
    """
    def __init__(self, in_channels, used_resid = True,num_repeat = 1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_repeat):
            self.layers+=[
                # Add block
                nn.Sequential(
                    CNNBlock(in_channels, in_channels // 2, kernel_size=1), # TODO: Check whether we add the layers correctly
                    CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1)
                )
            ]


        self.num_repeat =num_repeat
        self.used_resid = used_resid

    def forward(self, x):
        for block in self.layers:
            try:
                x = block(x) + x if self.used_resid else block(x)
            except :
                print("now")
                x = block(x) + x if self.used_resid else block(x)
        return x

class YOLO3(nn.Module):
    def __init__(self,in_channels = 3, num_classes = 20):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Now we want to create all the layers.
        
        self.layers = self.create_layer_config()

    def create_layer_config(self):
        """ 
        # tuple (out_cha, Kernel_size, stride)
        # list  [block, num_of _repeat]
        # string scale prediction / upsampling
        """
        in_channels = self.in_channels
        layers = nn.ModuleList()

        for layer in config:
            if(type(layer)==list):
                [block, num_of_repeat] = layer
                layers.append(ResidualBlock(
                    in_channels=in_channels,
                    num_repeat=num_of_repeat
                ))
            elif (type(layer)==tuple):
                out_channel , kernel_size, stride = layer
                layers.append(CNNBlock(in_channels,
                                       out_channel,
                                       kernel_size = kernel_size,
                                       stride = stride,
                                       padding =1 if kernel_size==3 else 0))
                in_channels = out_channel
            elif (type(layer) == str):
                if(layer =="S"):
                    layers += \
                    [
                        ResidualBlock(
                        in_channels,
                        used_resid=False,
                        num_repeat=1) #TODO: Check
                    ,

                        CNNBlock(in_channels=in_channels,
                                 out_channels= in_channels//2,kernel_size = 1)
                    ,
                        ScalePrediction(
                            in_channels=in_channels//2,
                            num_channels = self.num_classes)
                    ]
                    in_channels = in_channels//2

                elif (layer =="U"):
                    layers+= \
                        [
                        nn.Upsample(scale_factor=2)
                        ]
                    in_channels  = in_channels*3 # when we  have sampling layer, we concate. We should concat after the upsample
                    # We concate the one which was concat last.


        return layers
        

    def forward(self, x):
        print("YOLO Forward")

        output =[]
        route_connection =[] # for concat the channels

        for layer in self.layers:
            if(isinstance(layer, ScalePrediction)):
                output.append(layer(x))
                continue  # we continue so we would save the output yet we will come back to the root of the sequence.
            x = layer(x)

            if(isinstance(layer, ResidualBlock)) and layer.num_repeat==8:
                route_connection.append(x)

            elif (isinstance(layer, nn.Upsample)):
                x = torch.cat([x, route_connection[-1]], dim = 1) # along the channels
                route_connection.pop() # remove the last


        return output
if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLO3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print(type(out))
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")