import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu


import argparse

from bench_cpu_gpu import batch_size

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--eval', choices=['conv_ops','linear_ops','conv_network'], default='conv_network',
                    help='select the evaluation type: basic_ops, conv_ops, linear_ops, conv_network')

args = parser.parse_args()

batch = args.batch

def tput(model, name, input_shape=224, output_shape=1000, dev='cuda', input_channel=3):
    dev = xm.xla_device()
    with (torch.no_grad()):
        batchsize = batch
        print(f'batchsize: {batchsize}')
        model.eval()
        T = 0
        for i in range(150):
            input = torch.rand(batchsize, input_channel, input_shape, input_shape)
            input = input.to(dev)
            model = model.to(dev)
            start = time.time()
            y = model(input)
            end = time.time()
            s = y.sum()
            for param in model.parameters():
                param += s
                break
            if i > 50:
                T += (end - start)
        T /= 100
        #print(output.sum())
    print('Forward throughput: %10s : %6.3fms' % (name, 1000*T))

def tput_linear(model, name, input_channel, dev='cuda'):
    with (torch.no_grad()):
        dev = xm.xla_device()
        batchsize = batch
        print(f'batchsize: {batchsize}')
        model.eval()
        T = 0
        for i in range(150):
            input = torch.rand(batchsize, input_channel)
            time.sleep(0.05)
            input = input.to(dev)
            model = model.to(dev)
            start = time.time()
            y = model(input)
            end = time.time()
            s = y.sum()
            for param in model.parameters():
                param += s
                break
            if i > 50:
                T += (end - start)
        T /= 100
        #print(output.sum())
    print('Forward throughput: %10s : %6.3fms' % (name, 1000*T))


if __name__ == '__main__':
    print('Torchvision classification models test')
    print('ImageNet models')
    if args.eval == 'conv_network':
        print('ImageNet models')
        tput(models.alexnet(), 'alexnet')
        tput(models.mobilenet_v2(), 'mobilenet_v2')
        tput(models.mobilenet_v3_large(), 'mobilenet_v3_large')
        tput(models.mobilenet_v3_small(), 'mobilenet_v3_small')
        tput(models.mobilenet_v3_small(), 'mobilenet_v3_small')
        tput(models.vgg11_bn(), 'VGG11')
        tput(models.vgg13_bn(), 'VGG13')
        tput(models.vgg16_bn(), 'VGG16')
        tput(models.vgg19_bn(), 'VGG19')
        tput(models.resnet18(), 'resnet18')
        tput(models.resnet50(), 'resnet50')
        tput(models.resnet101(), 'resnet101')
        tput(models.resnet152(), 'resnet152')
        tput(models.densenet121(), 'densenet121')
        tput(models.densenet201(), 'densenet201')


        print('Cifar10 models')
        #tput(models.alexnet(num_classes=10), 'alexnet_cifar10', 32,10)
        tput(models.mobilenet_v2(num_classes=10), 'mobilenet_v2_cifar10', 32,10)
       # tput(models.mobilenet_v3_large(num_classes=10), 'mobilenet_v3_large_cifar10', 32,10)
       # tput(models.mobilenet_v3_small(num_classes=10), 'mobilenet_v3_small_cifar10', 32,10)
       # tput(models.mobilenet_v3_small(num_classes=10), 'mobilenet_v3_small_cifar10', 32,10)
        tput(models.vgg11_bn(num_classes=10), 'VGG11_cifar10', 32,10)
        tput(models.vgg13_bn(num_classes=10), 'VGG13_cifar10', 32,10)
        tput(models.vgg16_bn(num_classes=10), 'VGG16_cifar10', 32,10)
        tput(models.vgg19_bn(num_classes=10), 'VGG19_cifar10', 32,10)
        tput(models.resnet18(num_classes=10), 'resnet18_cifar10', 32,10)
        tput(models.resnet50(num_classes=10), 'resnet50_cifar10', 32,10)
        tput(models.resnet101(num_classes=10), 'resnet101_cifar10', 32,10)
        tput(models.resnet152(num_classes=10), 'resnet152_cifar10', 32,10)
        tput(models.densenet121(num_classes=10), 'densenet121_cifar10', 32,10)
        tput(models.densenet201(num_classes=10), 'densenet201_cifar10', 32,10)

        print('Cifar100 models')
        tput(models.mobilenet_v2(num_classes=100), 'mobilenet_v2_cifar100', 32,100)
        tput(models.mobilenet_v3_large(num_classes=100), 'mobilenet_v3_large_cifar100', 32,100)
        tput(models.mobilenet_v3_small(num_classes=100), 'mobilenet_v3_small_cifar100', 32,100)
        tput(models.mobilenet_v3_small(num_classes=100), 'mobilenet_v3_small_cifar100', 32,100)
        tput(models.vgg11_bn(num_classes=100), 'VGG11_cifar100', 32,100)
        tput(models.vgg13_bn(num_classes=100), 'VGG13_cifar100', 32,100)
        tput(models.vgg16_bn(num_classes=100), 'VGG16_cifar100', 32,100)
        tput(models.vgg19_bn(num_classes=100), 'VGG19_cifar100', 32,100)
        tput(models.resnet18(num_classes=100), 'resnet18_cifar100', 32,100)
        tput(models.resnet50(num_classes=100), 'resnet50_cifar100', 32,100)
        tput(models.resnet101(num_classes=100), 'resnet101_cifar100', 32,100)
        tput(models.resnet152(num_classes=100), 'resnet152_cifar100', 32,100)
        tput(models.densenet121(num_classes=100), 'densenet121_cifar100', 32,100)
        tput(models.densenet201(num_classes=100), 'densenet201_cifar100', 32,100)
    if args.eval == 'conv_ops':
        print('test conv2d (input size 224x224)')
        tput(nn.Conv2d(64, 64, 3, 1, 1), 'conv2d', input_channel = 64)
        tput(nn.Conv2d(256, 256, 3, 1, 1), 'conv2d' , input_channel = 256)
       # tput(nn.Conv2d(512, 512, 3, 1, 1), 'conv2d', input_channel = 512)

        print('test conv2d (input size 64 x 64)')
        tput(nn.Conv2d(64, 64, 3, 1, 1), 'conv2d', 64, input_channel=64)
        tput(nn.Conv2d(256, 256, 3, 1, 1), 'conv2d', 64, input_channel=256)
        tput(nn.Conv2d(512, 512, 3, 1, 1), 'conv2d', 64, input_channel=512)
        tput(nn.Conv2d(1024, 1024, 3, 1, 1), 'conv2d', 64, input_channel=1024)

        print('test conv2d (input size 32 x 32)')
        tput(nn.Conv2d(64, 64, 3, 1, 1), 'conv2d', 32, input_channel=64)
        tput(nn.Conv2d(256, 256, 3, 1, 1), 'conv2d', 32, input_channel=256)
        tput(nn.Conv2d(512, 512, 3, 1, 1), 'conv2d', 32, input_channel=512)
        tput(nn.Conv2d(1024, 1024, 3, 1, 1), 'conv2d', 32, input_channel=1024)

    if args.eval == 'linear_ops':
        print('test linear (input size 1024)')
        tput_linear(nn.Linear(512, 512), 'linear', 512)
        tput_linear(nn.Linear(1024, 1024), 'linear', 1024)
        tput_linear(nn.Linear(2048, 2048), 'linear', 2048)
        tput_linear(nn.Linear(4096, 4096), 'linear', 4096)
        tput_linear(nn.Linear(8192, 8192), 'linear', 8192)
