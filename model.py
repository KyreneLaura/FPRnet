import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import math
from torch.nn import functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:    
        m.weight.data = init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')      
    elif classname.find('Linear') != -1:      
        m.weight.data = init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        m.bias.data.fill_(0)   
    elif classname.find('BatchNorm1d') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)    

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data = init.normal_(m.weight.data,  0, 0.001)
        m.bias.data.fill_(0)
       

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
 
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)  
        classifier.apply(weights_init_classifier)
        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x



# RGB
class visible_net_FPN_SRCNN(nn.Module):
    def __init__(self, arch='resnet18'):
        super(visible_net_FPN_SRCNN, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
    
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear',align_corners=True) + y

    def forward(self, x):
        c1 = self.visible.conv1(x)
        c1 = self.visible.bn1(c1)
        c1 = self.visible.relu(c1)
        c1 = self.visible.maxpool(c1)
        c2 = self.visible.layer1(c1)
        c3 = self.visible.layer2(c2)
        c4 = self.visible.layer3(c3)
        c5 = self.visible.layer4(c4)
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        c5 = self.visible.avgpool(c5)
        p5 = self.visible.avgpool(p5)
        p4 = self.visible.avgpool(p4)
        p3 = self.visible.avgpool(p3)
        p2 = self.visible.avgpool(p2)
        y = c5
        c5 = c5.view(c5.size(0), c5.size(1))
        p5 = p5.view(p5.size(0), p5.size(1))
        p3 = p3.view(p3.size(0), p3.size(1))
        p4 = p4.view(p4.size(0), p4.size(1))
        p2 = p2.view(p2.size(0), p2.size(1)) 
        return y, c5, p5,p4,p2
     
# IR
class thermal_net_FPN_SRCNN(nn.Module):
    def __init__(self, arch='resnet18'):
            super(thermal_net_FPN_SRCNN, self).__init__()
            if arch == 'resnet18':
                model_ft = models.resnet18(pretrained=True)
            elif arch == 'resnet50':
                model_ft = models.resnet50(pretrained=True)
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.thermal = model_ft
            self.dropout = nn.Dropout(p=0.5)
         
            self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  
            self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
            self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
            '''Upsample and add two feature maps.
            Args:
              x: (Variable) top feature map to be upsampled.
              y: (Variable) lateral feature map.
            Returns:
              (Variable) added feature map.
            Note in PyTorch, when input size is odd, the upsampled feature map
            with `F.upsample(..., scale_factor=2, mode='nearest')`
            maybe not equal to the lateral feature map size.
            e.g.
            original input size: [N,_,15,15] ->
            conv2d feature map size: [N,_,8,8] ->
            upsampled feature map size: [N,_,16,16]
            So we choose bilinear upsample which supports arbitrary output sizes.
            '''
            _, _, H, W = y.size()
            return F.upsample(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):      
            c1 = self.thermal.conv1(x)
            c1 = self.thermal.bn1(c1)
            c1 = self.thermal.relu(c1)
            c1 = self.thermal.maxpool(c1)   
            c2 = self.thermal.layer1(c1)
            c3 = self.thermal.layer2(c2)
            c4 = self.thermal.layer3(c3)
            c5 = self.thermal.layer4(c4)
    
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p2 = self._upsample_add(p3, self.latlayer3(c2))
        
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)
            
            c5 = self.thermal.avgpool(c5)
            p5 = self.thermal.avgpool(p5)
            p4 = self.thermal.avgpool(p4)
            p3 = self.thermal.avgpool(p3)
            p2 = self.thermal.avgpool(p2)

            y = c5
            c5 = c5.view(c5.size(0), c5.size(1))
            p5 = p5.view(p5.size(0),p5.size(1))
            p4 = p4.view(p4.size(0),p4.size(1))
            p3 = p3.view(p3.size(0),p3.size(1))
            p2 = p2.view(p2.size(0),p2.size(1))
            return y,c5,p5,p4,p2
           


class embed_net(nn.Module):
    # low_dim = 512
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50', FPN = False):
        super(embed_net, self).__init__()
        self.FPN = FPN
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
        elif arch =='resnet50':
            if FPN:
                print("FPN",FPN)
                self.visible_net = visible_net_FPN_SRCNN(arch=arch)
                self.thermal_net = thermal_net_FPN_SRCNN(arch=arch)
                pool_dim1 = 256
                self.feature_FPN = FeatureBlock(pool_dim1,128 , dropout=drop)
                self.classifier_FPN = ClassBlock(64, class_num, dropout=drop)
               
            pool_dim = 2048
        self.feature = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout = drop)
        self.l2norm = Normalize(2) 
        self.fc1 = FeatureBlock(128,64, dropout = drop)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self, x1, x2, modal = 0 ):
        if modal==0:
            if self.FPN:   
               y1, x1, p1, q1, k1 = self.visible_net(x1)
               y2, x2, p2, q2, k2 = self.thermal_net(x2)
               x = torch.cat((x1, x2), 0)   
               p = torch.cat((p1, p2),0)
               q = torch.cat((q2, q1),0)
               k = torch.cat((k2, k1), 0)
               t1 = torch.cat((p, q), 0)
               t2 = torch.cat((p, k), 0)
               m = torch.cat((t1, t2), 0)
        elif modal ==1:
            if self.FPN: 
               y1, x, p, q, k = self.visible_net(x1)     
               t1 = torch.cat((p, q), 0)
               t2 = torch.cat((p, k), 0)
               m = torch.cat((t1, t2), 0)      
        elif modal ==2:
            if self.FPN:
                y2, x, p, q, k = self.thermal_net(x2)
                t1 = torch.cat((p, q), 0)
                t2 = torch.cat((p, k), 0)
                m = torch.cat((t1, t2), 0)


        y = self.feature(x)
        y = self.leaky(y)
        out1 = self.classifier(y)
        if self.FPN:       
            z2 = self.feature_FPN(m)
            z2 = self.relu(z2) 
            z2 = self.fc1(z2)
            z2 = self.relu(z2)
            out2 = self.classifier_FPN(z2)   
        if self.training:
            _, _, H, W = y1.size()
            y1 = F.upsample(y1, size=(H, W), mode='bilinear', align_corners=True)
            y1 = y1.view(y1.size(0), y1.size(1))
            _, _, H, W = y2.size()
            y2 = F.upsample(y2, size=(H, W), mode='bilinear', align_corners=True)
            y2 = y2.view(y2.size(0), y2.size(1))
            yRGB = self.feature(y1)
            yIR = self.feature(y2)
            outRGB = self.classifier(yRGB)
            outIR = self.classifier(yIR)
            if self.FPN:    
                 return out1, out2, outRGB, outIR,self.l2norm(y), self.l2norm(z2), self.l2norm(yRGB), self.l2norm(yIR)          
        else:
            if self.FPN:
                return self.l2norm(x),self.l2norm(m),self.l2norm(y),self.l2norm(z2)
            
