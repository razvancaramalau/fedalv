import torch
import torch.nn.functional as F
from .cnn4conv import CNN4Conv
from .mobilenet import MobileNetCifar
from .resnet import *
from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import torch.nn as nn
import torch.distributions as distributions


class ModelBase(nn.Module):
    def __init__(self, args, back_bone, z_dim, probabilistic):
        super(ModelBase, self).__init__()
        self.probabilistic = probabilistic
        self.z_dim = z_dim
        self.num_samples = args.num_samples
        self.num_classes = args.num_classes
        out_dim = 2 * self.z_dim if self.probabilistic else self.z_dim
        if back_bone == 'resnet18':
            net = resnet18(weights=ResNet18_Weights.DEFAULT)
            net.fc = nn.Linear(net.fc.in_features, out_dim)
        elif back_bone == 'resnet50':
            net = resnet50(weights=ResNet50_Weights.DEFAULT)
            net.fc = nn.Linear(net.fc.in_features, out_dim)
        else:
            raise NotImplementedError
        self.net = net
        self.cls = nn.Linear(z_dim, args.num_classes)
        self.model = nn.Sequential(self.net, self.cls)
        
    def featurize(self,x,num_samples=1,return_dist=False):
        if not self.probabilistic:
            return self.net(x)
        else:
            z_params = self.net(x)
            z_mu = z_params[:,:self.z_dim]
            z_sigma = F.softplus(z_params[:,self.z_dim:])
            # keep z_mu / z_sigma > 0 
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma),1)
            z = z_dist.rsample([num_samples]).view([-1,self.z_dim])
            
            if return_dist:
                return z, (z_mu,z_sigma)
            else:
                return z
            
    def get_embedding_dim(self):
        return self.z_dim
    
    def forward(self, x):
        if not self.probabilistic:
            return self.model(x), self.net(x)
        else:
            if self.training:
                z = self.featurize(x)
                return self.cls(z)
            else:
                z = self.featurize(x,num_samples=self.num_samples)
                # preds = torch.softmax(self.cls(z),dim=1)
                preds = self.cls(z)
                # _, preds = torch.min(predict, 1)
                preds = preds.view([self.num_samples,-1,self.num_classes]).mean(0)
                # return torch.log(preds), self.net(x)
                return preds, self.net(x)
            
def get_model(args):
    probabilistic = True if (args.fl_algo in ['feda', 'fedaga']) else False
    if args.model == 'cnn4conv':
        net_glob = CNN4Conv(in_channels=args.in_channels, num_classes=args.num_classes, args=args).to(args.device)
    elif args.model == 'mobilenet':
        net_glob = MobileNetCifar(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet10':
        net_glob = resnet10(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet18':
        # net_glob = resnet18(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
        out_dim = 512
        # net_glob = resnet18(weights=ResNet18_Weights.DEFAULT)
        # net_glob.fc = nn.Linear(net_glob.fc.in_features, out_dim)
        # cls = nn.Linear(out_dim, args.num_classes)
        # net_glob = nn.Sequential(net_glob, cls)
        
        net_glob = ModelBase(args, 'resnet18', out_dim, probabilistic)
        # net_glob.probabilistic = False
        
        
        
    elif args.model == 'resnet50':
        # net_glob = resnet50(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
        out_dim = 2048
        # net_glob = resnet50(weights=ResNet50_Weights.DEFAULT)
        # net_glob.fc = nn.Linear(net_glob.fc.in_features,out_dim)
        # cls = nn.Linear(out_dim,args.num_classes)
        # net_glob = nn.Sequential(net_glob, cls)
        
        net_glob = ModelBase(args, 'resnet50', out_dim, probabilistic)
    else:
        exit('Error: unrecognized model')
    # print(net_glob)
    net_glob.to(args.device)
    return net_glob