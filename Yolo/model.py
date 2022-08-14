import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

architecture_config = [
    # tuple (kernelSize, num_filters,stride, padding/
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",

    #kernelSize, num_filters,stride, padding
    #list  (tuples, numofRepeat)

    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(CNNBlock, self).__init__()
        # print("CNNBlock")

        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs).to(device)
        self.batchNorm = nn.BatchNorm2d(out_channels).to(device) # with learnable
        self.leakyLu = nn.LeakyReLU(0.1)

    def forward(self, x):
        self.leakyLu(self.batchNorm(self.conv(x))).to(device)

class Yolo1(nn.Module):
    def __init__(self, im_channels = 3, **kwargs):
        super(Yolo1,self).__init__()
        # print("Yolo")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.architecture = architecture_config
        self.in_channels = im_channels

        self.darknet = self.create_conv_layers(self.architecture).to(device)
        self.fcs = self.create_fcs(**kwargs).to(device)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim = 1))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for idx, x in enumerate(architecture):
            if(type(x) is tuple):
                (kernelSize, num_filters, stride, padding) = x
                layers += [
                    CNNBlock(in_channels =in_channels ,
                                   out_channels =kernelSize,
                                    kernel_size= num_filters,
                                   stride=stride,
                                   padding=padding)
                           ]
                in_channels = x[1]
            elif(type(x) =="M"):
                layers+= [
                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                ]
            elif (type(x)==list):
                assert  len(x) ==3
                conv_1 = x[0]
                conv_2 = x[1]
                num_repeat = x[2] # integer with the amount of times we repeat the process

                for _ in range(num_repeat):
                    layers +=[
                        # kernelSize, num_filters,stride, padding
                        CNNBlock(in_channels, # in channels
                                 conv_1[1], # out channels is the input size of the following layer
                                 kernel_size = conv_1[0],
                                 stride = conv_1[2],
                                 padding = conv_1[3]
                                 )
                    ]
                    layers +=[
                        CNNBlock(conv_1[1], # the input size is the output size of the previous layer
                             conv_2[1],
                             kernel_size=conv_2[0],
                             stride=conv_2[2],
                             padding=conv_2[3]
                             )
                    ]
                    in_channels = conv_2[1] # so when we run the loop once again, we will make sure we use the previous layer output (of the list in the sequence)

        return nn.Sequential(*layers)



    def create_fcs(self,split_size, num_boxes, num_classes):
        S,B,C = split_size, num_boxes, num_classes
        return nn.Sequential\
                (
            nn.Flatten(),
            nn.Linear(1024*S*S, 496),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5))
        )



def test( S = 7,  B =2 , C =20):
    print("test")
    model =Yolo1(split_size = S,num_boxes = B, num_classes = C).to(device)
    x = torch.randn((2, 3, 448, 448)).to(device)
    print("{model(x).shape=}")
    print(model(x).shape)


if __name__=='__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count() )
    print(torch.version.cuda)

    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    test()

