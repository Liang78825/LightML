# LightML

## Paper Abstract
The rapid integration of AI technologies into everyday life across sectors such as healthcare, autonomous driving, and smart home applications requires extensive computational resources, straining server infrastructure and incurring significant costs.

We present LightML, the first system-level photonic crossbar design, optimized for high-performance machine learning applications. This work provides the first complete memory and buffer architecture carefully designed to support the high-speed photonic crossbar, achieving more than 80% utilization. LightML also introduces solutions for key ML functions, including large-scale matrix multiplication (MMM), element-wise operations, non-linear functions, and convolutional layers. Delivering 320 TOP/s at only 3 watts, LightML offers significant improvements in speed and power efficiency, making it ideal for both edge devices and dense data center workloads.

## Artifact Abstract
This artifact page consists of two parts: a Cycle-Accurate Simulator, a PSpice Analog Circuit. Part I. The cycle-accurate simulator evaluates the latency for LightML with three catogories of task -- basic opeartion, convolutional network, and attention network. We also implement the equvialent task on the baseline machine -- CPU, GPU, and TPU, and with the measurement of their correspond latency; Part II. The PSPice analog circuit implements the circuit of each unit cell in the optical crossbar. The implementation provide the detail of the component including the RC-circuit, and the ADC reader.

### Artifact Repository: https://doi.org/10.5281/zenodo.15079938


## Part I: Cycle-Accurate Simulator

### Repository Structure
The repository for this part is structured as follows:
```
Benchmark_simulator
|__ bench_conv.py           # Benchmark evaluation on the convoluitional network
|__ bench_cpu_gpu.py        # Benchmark evaluation in the CPU / GPU devices
|__ bench_lightML.py        # Benchmark evaluation in the LightML
|__ bench_tpu.py            # Benchmark evaluation in the TPU devices
```

### Environment Requirement 

The evaluation for the baseline devices is built on the ``` PyTorch ``` envirnoment. CPU and GPU enviroment can be built following the offical instruction: https://pytorch.org/. The Google TPU is implement via the ``` gcloud ``` service, which can be allocated via this link: https://cloud.google.com/sdk?hl=en. The setup for the ```PyTorch``` environment can be referred to: https://cloud.google.com/tpu/docs/run-calculation-pytorch. 

### A. Usage -- ``` bench_lightML.py ```

Type ``` python bench_lightML.py -h``` in the command line to access the help page:

```
usage: bench_lightML.py [-h] [-b BATCH_SIZE] [-p] [-e {basic_ops,conv_networks,attention_networks}]

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size for the simulation
  -p, --print_utilization
                        print the utilization of the crossbar
  -e {basic_ops,conv_networks,attention_networks}, --evaluate {basic_ops,conv_networks,attention_networks}
                        choose the type of network to evaluate: basic_ops, conv_networks, attention_networks

```

* Example: ``` python bench_lightML.py -b 32 -p -e 'conv_networks' ``` starts the simulation with batch is 32, print the ultization detail, and test on the convolutional networks benchmark

* The expect output will be:
```
<...>
Evaluating the model:  VGG19
The utilization for the memory read is:  0.3497340726656211
The utilization for the memory write is:  0.07082666475770699
The utilization for the crossbar is:  0.5341034183769996
The utilization for the ADC is:  0.22008089216593632
The total runtime is:  5475932.0
The total cycle for the VGG19 is:  5475932.0
The total time for the VGG19 is:  0.4563276666666667 ms

<...>
```

### B. Usage -- ``` bench_cpu_gpu.py ```
Type ``` python bench_cpu_gpu.py -h``` to access the help page:
```
usage: bench_cpu_gpu.py [-h] [--device {cuda,cpu}] [--batch BATCH] [--eval {basic_ops,conv_ops,linear_ops,conv_network}]

options:
  -h, --help            show this help message and exit
  --device {cuda,cpu}   Device to run the benchmark on cuda or CPU (default: cuda)
  --batch BATCH         Batch size (default: 32)
  --eval {basic_ops,conv_ops,linear_ops,conv_network}
                        select the evaluation type: basic_ops, conv_ops, linear_ops, conv_network

```
* Example: ``` python bench_cpu_gpu.py --device 'cuda' --eval 'conv_network'``` to start the evaluation on cuda environment to test the convolutional network benchmarks

* The expect output will be:
```
Forward throughput:    alexnet :  1.314ms
batchsize: 32
Forward throughput: mobilenet_v2 :  4.258ms
batchsize: 32
Forward throughput: mobilenet_v3_large :  3.476ms
batchsize: 32
Forward throughput: mobilenet_v3_small :  3.065ms
batchsize: 32
<...>
```

## Part II: PSPice Circuit

### Software Installation:

The PSPice circuit is implemented in the Cadence OrCAD environment, which can be accessed via https://www.cadence.com/en_US/home/tools/pcb-design-and-analysis/orcad.html.


