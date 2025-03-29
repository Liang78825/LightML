import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.cuda import Event

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cuda','cpu'], default='cuda', help='Device to run the benchmark on cuda or CPU (default: cuda)')
parser.add_argument('--batch', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--eval', choices=['basic_ops','conv_ops','linear_ops','conv_network'], default='basic_ops',
                    help='select the evaluation type: basic_ops, conv_ops, linear_ops, conv_network')


args = parser.parse_args()
batch_size = args.batch
device = args.device

if torch.cuda.is_available():
    print('Cuda avilable', torch.cuda.get_device_name(0))
else:
    print('cuda is not avilable')
    device = 'cpu'




def tput(model, name, input_shape=224, output_shape=1000, dev='cuda', input_channel=3):
    with (torch.no_grad()):
        batchsize = batch_size
        print(f'batchsize: {batchsize}')
        model.eval()
        T = 0
        for i in range(20):
            input = torch.rand(batchsize, input_channel, input_shape, input_shape)
            if torch.cuda.is_available() and dev == 'cuda':
                input = input.to('cuda')
                model = model.to('cuda')
                start, end = Event(True), Event(True)

                torch.cuda.synchronize()
                start.record()
                y = model(input)
                end.record()
                torch.cuda.synchronize()
                if i > 5:
                    T += start.elapsed_time(end)

            else:
                start = time.time()
                y = model(input)
                end = time.time()
                if i > 5:
                    T += (end - start)
        T /= 15
        #print(output.sum())
    print('Forward throughput: %10s : %6.3fms' % (name, T))

def tput_linear(model, name, input_channel, dev='cuda'):
    with (torch.no_grad()):
        batchsize = batch_size
        print(f'batchsize: {batchsize}')
        model.eval()
        T = 0
        for i in range(150):
            input = torch.rand(batchsize, input_channel)
            time.sleep(0.05)
            if torch.cuda.is_available() and dev == 'cuda':
                input = input.to('cuda')
                model = model.to('cuda')
                start, end = Event(True), Event(True)
                torch.cuda.synchronize()
                start.record()
                y = model(input)
                end.record()
                end.synchronize()
                if i > 50:
                    T += start.elapsed_time(end)
            else:
                start = time.time()
                y = model(input)
                end = time.time()
                if i > 50:
                    T += (end - start)
        T /= 100
        #print(output.sum())
    print('Forward throughput: %10s : %6.3fms' % (name, T))


def tput_add(input_shape=224, dev='cuda'):
    with (torch.no_grad()):
        batchsize = batch_size
        print(f'batchsize: {batchsize}')
        T = 0
        input = torch.rand(batchsize, input_shape)
        input = input.to('cpu')
        if torch.cuda.is_available() and dev == 'cuda':
            input = input.to('cuda')
            torch.cuda.synchronize()

        #torch.cuda.synchronize()
        start = time.time()
        for i in range(100):
            input = input * 0.5

        if torch.cuda.is_available() and dev == 'cuda':
           torch.cuda.synchronize()
        print(input.sum())
        end = time.time()
        T = end - start
        T/=100
        #print(output.sum())
    print('Forward throughput: %10s : %6.3fms' % ('add', 1000*T))


def tput_mul(input_shape=224, dev='cuda'):
    with (torch.no_grad()):
        batchsize = batch_size
        print(f'batchsize: {batchsize}')
        T = 0
        for i in range(150):
            input = torch.rand(batchsize, input_shape)
            if torch.cuda.is_available() and dev == 'cuda':
                input = input.to('cuda')
                start, end = Event(True), Event(True)
                torch.cuda.synchronize()
                start.record()
                input = input + input
                end.record()
                torch.cuda.synchronize()
                if i > 50:
                    T += start.elapsed_time(end)
            else:
                start = time.time()
                input = input + input
                end = time.time()
                if i > 50:
                    T += (end - start)
        T /= 100
        #print(output.sum())
    print('Forward throughput: %10s : %6.3fms' % ('mul', 1000*T))

def tput_scale(input_shape=224, dev='cuda'):
    with (torch.no_grad()):
        batchsize = batch_size
        print(f'batchsize: {batchsize}')
        T = 0
        for i in range(150):
            if torch.cuda.is_available() and dev == 'cuda':
                input = torch.rand(batchsize, input_shape)
                input = input.to('cuda')
                start, end = Event(True), Event(True)
                torch.cuda.synchronize()
                start.record()
                input = input * 0.5
                end.record()
                torch.cuda.synchronize()
                if i > 50:
                    T += start.elapsed_time(end)
            else:
                start = time.time()
                input = input * 0.5
                end = time.time()
                if i > 50:
                    T += (end - start)
        T /= 100
        #print(output.sum())
    print('Forward throughput: %10s : %6.3fms' % ('scale', T))

if __name__ == '__main__':
    print('Torchvision classification models test')
    print('ImageNet models')
    if args.eval == 'basic_ops':
        # test elementwise operation, e.g. add, mul, div, sub
        print('adding operation')
        tput_add(1024*1024, device)
        tput_add(2048*2048, device)
        tput_add(4096*4096, device)

        print('multiplication operation')
        tput_mul(1024*1024, device)
        tput_mul(2048*2048, device)
        tput_mul(4096*4096, device)

        print('scaling operation')
        tput_scale(1024*1024, device)
        tput_scale(2048*2048, device)
        tput_scale(4096*4096, device)

    if False:
        model_fp32 = models.resnet50(pretrained=True)

        # Step 2: Specify quantization configuration
        # We'll use per-channel weight quantization (default is per-tensor)
        model_fp32.eval()
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Step 3: Fuse model layers
        # Some layers like Conv, BatchNorm, and ReLU can be fused to optimize performance
        model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'bn1'],
                                                                        ['layer1.0.conv1', 'layer1.0.bn1'
                                                            ],
                                                                        ['layer1.0.conv2', 'layer1.0.bn2'],
                                                                        ['layer1.1.conv1', 'layer1.1.bn1'],
                                                                        ['layer1.1.conv2', 'layer1.1.bn2']])

        # Step 4: Prepare for quantization
        model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

        # Step 5: Calibrate the model with a few input samples (this step is optional)
        # Here we just run a few inference passes to simulate calibration
        dummy_input = torch.randn(1, 3, 224, 224)
        model_fp32_prepared(dummy_input)

        # Step 6: Convert to quantized model
        model_int8 = torch.quantization.convert(model_fp32_prepared)

        # Step 7: Test the quantized model
        # Now we can test the quantized model with new data

        batch_t = torch.randn(1, 3, 224, 224)

        # Perform inference using the quantized model
        with torch.no_grad():
            model_int8.eval()
            output = model_int8(batch_t)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        print(f"Predicted class: {predicted_class.item()}")

    if args.eval == 'conv_network':
        print('ImageNet dataset')
        tput(models.alexnet(), 'alexnet',dev=device)
        tput(models.mobilenet_v2(), 'mobilenet_v2',dev=device)
        tput(models.mobilenet_v3_large(), 'mobilenet_v3_large',dev=device)
        tput(models.mobilenet_v3_small(), 'mobilenet_v3_small',dev=device)
        tput(models.mobilenet_v3_small(), 'mobilenet_v3_small',dev=device)
        tput(models.vgg11_bn(), 'VGG11',dev=device)
        tput(models.vgg13_bn(), 'VGG13',dev=device)
        tput(models.vgg16_bn(), 'VGG16',dev=device)
        tput(models.vgg19_bn(), 'VGG19',dev=device)
        tput(models.resnet18(), 'resnet18',dev=device)
        tput(models.resnet50(), 'resnet50',dev=device)
        tput(models.resnet101(), 'resnet101',dev=device)
        tput(models.resnet152(), 'resnet152',dev=device)
        tput(models.densenet121(), 'densenet121',dev=device)
        tput(models.densenet201(), 'densenet201',dev=device)

        print('Cifar10 dataset')
        #tput(models.alexnet(num_classes=10), 'alexnet_cifar10', 32,10)
        tput(models.mobilenet_v2(num_classes=10), 'mobilenet_v2_cifar10', 32,10,dev=device)
        tput(models.mobilenet_v3_large(num_classes=10), 'mobilenet_v3_large_cifar10', 32,10,dev=device)
        tput(models.mobilenet_v3_small(num_classes=10), 'mobilenet_v3_small_cifar10', 32,10,dev=device)
        tput(models.mobilenet_v3_small(num_classes=10), 'mobilenet_v3_small_cifar10', 32,10,dev=device)
        tput(models.vgg11_bn(num_classes=10), 'VGG11_cifar10', 32,10,dev=device)
        tput(models.vgg13_bn(num_classes=10), 'VGG13_cifar10', 32,10,dev=device)
        tput(models.vgg16_bn(num_classes=10), 'VGG16_cifar10', 32,10,dev=device)
        tput(models.vgg19_bn(num_classes=10), 'VGG19_cifar10', 32,10,dev=device)
        tput(models.resnet18(num_classes=10), 'resnet18_cifar10', 32,10,dev=device)
        tput(models.resnet50(num_classes=10), 'resnet50_cifar10', 32,10,dev=device)
        tput(models.resnet101(num_classes=10), 'resnet101_cifar10', 32,10,dev=device)
        tput(models.resnet152(num_classes=10), 'resnet152_cifar10', 32,10,dev=device)
        tput(models.densenet121(num_classes=10), 'densenet121_cifar10', 32,10,dev=device)
        tput(models.densenet201(num_classes=10), 'densenet201_cifar10', 32,10,dev=device)

        print('Cifar100 datasets')
        tput(models.mobilenet_v2(num_classes=100), 'mobilenet_v2_cifar100', 32,100,dev=device)
        tput(models.mobilenet_v3_large(num_classes=100), 'mobilenet_v3_large_cifar100', 32,100,dev=device)
        tput(models.mobilenet_v3_small(num_classes=100), 'mobilenet_v3_small_cifar100', 32,100,dev=device)
        tput(models.mobilenet_v3_small(num_classes=100), 'mobilenet_v3_small_cifar100', 32,100,dev=device)
        tput(models.vgg11_bn(num_classes=100), 'VGG11_cifar100', 32,100,dev=device)
        tput(models.vgg13_bn(num_classes=100), 'VGG13_cifar100', 32,100,dev=device)
        tput(models.vgg16_bn(num_classes=100), 'VGG16_cifar100', 32,100,dev=device)
        tput(models.vgg19_bn(num_classes=100), 'VGG19_cifar100', 32,100,dev=device)
        tput(models.resnet18(num_classes=100), 'resnet18_cifar100', 32,100,dev=device)
        tput(models.resnet50(num_classes=100), 'resnet50_cifar100', 32,100,dev=device)
        tput(models.resnet101(num_classes=100), 'resnet101_cifar100', 32,100,dev=device)
        tput(models.resnet152(num_classes=100), 'resnet152_cifar100', 32,100,dev=device)
        tput(models.densenet121(num_classes=100), 'densenet121_cifar100', 32,100,dev=device)
        tput(models.densenet201(num_classes=100), 'densenet201_cifar100', 32,100,dev=device)
    if args.eval == 'conv_ops':
        print('test conv2d (input size 224x224)')
        tput(nn.Conv2d(64, 64, 3, 1, 1), 'conv2d', input_channel = 64,dev=device)
        tput(nn.Conv2d(256, 256, 3, 1, 1), 'conv2d' , input_channel = 256,dev=device)
        tput(nn.Conv2d(512, 512, 3, 1, 1), 'conv2d', input_channel = 512,dev=device)

        print('test conv2d (input size 64 x 64)')
        tput(nn.Conv2d(64, 64, 3, 1, 1), 'conv2d', 64, input_channel=64,dev=device)
        tput(nn.Conv2d(256, 256, 3, 1, 1), 'conv2d', 64, input_channel=256,dev=device)
        tput(nn.Conv2d(512, 512, 3, 1, 1), 'conv2d', 64, input_channel=512,dev=device)
        tput(nn.Conv2d(1024, 1024, 3, 1, 1), 'conv2d', 64, input_channel=1024,dev=device)

        print('test conv2d (input size 32 x 32)')
        tput(nn.Conv2d(64, 64, 3, 1, 1), 'conv2d', 32, input_channel=64,dev=device)
        tput(nn.Conv2d(256, 256, 3, 1, 1), 'conv2d', 32, input_channel=256,dev=device)
        tput(nn.Conv2d(512, 512, 3, 1, 1), 'conv2d', 32, input_channel=512,dev=device)
        tput(nn.Conv2d(1024, 1024, 3, 1, 1), 'conv2d', 32, input_channel=1024,dev=device)

    if args.eval == 'linear_ops':
        print('test linear (input size 1024)')
        tput_linear(nn.Linear(512, 512), 'linear', 512,dev=device)
        tput_linear(nn.Linear(1024, 1024), 'linear', 1024,dev=device)
        tput_linear(nn.Linear(2048, 2048), 'linear', 2048,dev=device)
        tput_linear(nn.Linear(4096, 4096), 'linear', 4096,dev=device)
        tput_linear(nn.Linear(8192, 8192), 'linear', 8192,dev=device)
