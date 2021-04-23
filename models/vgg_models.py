import torch
from torch import nn
from general_utils import reproducible_state
from torchsummary import summary
from math import pow



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self,vgg_name,seed=30,num_classes=10,input_height = 32):
        super(VGG, self).__init__()
        reproducible_state(seed=seed)
        self.features = self._make_layers(cfg[vgg_name])

        # expanding the linear layer ( in the case of tiny_imageNet)
        linear_expansion_rate = int(pow(input_height/32,2))
        self.classifier = nn.Linear(linear_expansion_rate *512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class VGG_Intermediate_Branches(nn.Module):
    def __init__(self,vgg_name,seed=30,num_classes=10,input_height = 32):
        super(VGG_Intermediate_Branches, self).__init__()
        reproducible_state(seed=seed)
        self.vgg_name= vgg_name
        features = self._make_layers(cfg[vgg_name])
        if self.vgg_name == "VGG16":
            self.features_1 = features[:3]
            self.features_2 = features[3:6]
            self.features_3 = features[6:10]
            self.features_4 = features[10:14]
            self.features_5 = features[14:]
        else:

            self.features_1 = features[:4]
            self.features_2 = features[4:8]
            self.features_3 = features[8:15]
            self.features_4 = features[15:22]
            self.features_5 = features[22:]
            self.auxiliary1 =self._make_layers(cfg[vgg_name][2:])
            self.auxiliary2 =self._make_layers(cfg[vgg_name][4:])
            self.auxiliary3 =self._make_layers(cfg[vgg_name][7:])
            self.auxiliary4 =self._make_layers(cfg[vgg_name][10:])



        # expanding the linear layer ( in the case of tiny_imageNet)
        linear_expansion_rate = int(pow(input_height / 32, 2))
        self.classifier_1 = nn.Linear(linear_expansion_rate *512, num_classes)
        self.classifier_2 = nn.Linear(linear_expansion_rate *512, num_classes)
        self.classifier_3 = nn.Linear(linear_expansion_rate *512, num_classes)
        self.classifier_4 = nn.Linear(linear_expansion_rate *512, num_classes)
        self.classifier_5 = nn.Linear(linear_expansion_rate *512, num_classes)

        #self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):

        feature_list = []
        out_1 = self.features_1(x)
        feature_list.append(out_1)
        out_2 = self.features_2(out_1)
        feature_list.append(out_2)

        out_3 = self.features_3(out_2)
        feature_list.append(out_3)

        out_4 = self.features_4(out_3)
        feature_list.append(out_4)

        out5_feature = self.features_5(out_4)



        out1_feature = self.auxiliary1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(x.size(0), -1)
        out4_feature = self.auxiliary4(feature_list[3]).view(x.size(0), -1)

        out1 = self.classifier_1(out1_feature)
        out2 = self.classifier_2(out2_feature)
        out3 = self.classifier_3(out3_feature)
        out4 = self.classifier_4(out4_feature)
        out5 = self.classifier_5(out5_feature)

        return [out5,out4,out3, out2, out1], [out4_feature,out3_feature, out2_feature, out1_feature]



















        outputs_list = []
        out_1 = self.features_1(x)
        outputs_list.append(out_1)
        out_2 = self.features_2(out_1)
        outputs_list.append(out_2)

        out_3 = self.features_3(out_2)
        outputs_list.append(out_3)

        out_4 = self.features_4(out_3)
        outputs_list.append(out_4)



        out = out_3.view(out_3.size(0), -1)


        final_out = self.classifier(out)
        outputs_list.append(final_out)

        return outputs_list

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)




def test():
    net = VGG_Intermediate_Branches('VGG11',num_classes=100)
    x = torch.randn(1,3,32,32)
    #y = net(x)
    summary(net,input_size=(3,32,32),device="cpu")
    #for out in y:
       # print("Out Size ",out.size())

    print("*"*40)
    from general_utils import intermediate_ouput_sizer,get_intermediate_ouput_dict

    intermediate_dict = get_intermediate_ouput_dict(model=net,model_module=net.features_1,example_input_tensor=x)

    for (key,value) in intermediate_dict.items():

        print(key,"\t\t",value.size())

test()

#test = Ablation_VGG_Intermediate_Branches("VGG11",seed=3,num_classes=100)
#spatial_size=32
#test = VGG_Intermediate_Branches("VGG11",seed=3,num_classes=100,input_height=spatial_size)
#input = torch.randn(1,3,spatial_size,spatial_size)
#summary(test,input_size=(3,spatial_size,spatial_size),device="cpu")
#y = test(input)
#for out in y:
    #print("Out shape ", out.shape)

