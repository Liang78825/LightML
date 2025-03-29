import numpy as np
import math

from click.core import batch
from six import print_


import argparse




parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=16, help='batch size for the simulation')
parser.add_argument('-p','--print_utilization', action='store_true', default= False,
                    help='print the utilization of the crossbar')
parser.add_argument('-e', '--evaluate', choices=['basic_ops', 'conv_networks','attention_networks'],
                    default='conv_network', help='choose the type of network to evaluate: basic_ops, conv_networks, attention_networks')



# simulate a ml accerlerator
# specs for the crossbar
crossbar_size = 128
crossbar_frequency = 12  # GHz
max_pulse = 128 * 9  # pulse
crossbar_cycle = 1  # cycle
crossbar_element_wise_cycle = 10 # cycle

# specs for the memory
memory_bandwidth = 900  # GB/s
memory_frequency = 4  # GHz=
memory_cycle = crossbar_cycle * crossbar_frequency / memory_frequency  # cycle
memory_address_look_cycle = 32 * memory_cycle  # cycle
memory_bandwidth_per_cycle = memory_bandwidth / memory_frequency  # GB e.g. (900GB/s / 4GHz = 225B/cycle)

# specs for the readout unit
readout_frequency = 1  # GHz
reader_row_size = crossbar_size  # row
reader_batch = 4  # batch
readout_cycle = crossbar_cycle * crossbar_frequency / readout_frequency  # cycle



# specs for the precision
input_bit_precision = 12
readout_bit_precision = 8

# specs for the on-chip cache
input_cache_size_max = 64 * 1024 * 2 # 128KB (double buffer)
weight_cache_size_max = 64 * 1024 * 2 # 128KB (double buffer)
output_cache_size_max = 64 * 1024  # 64KB
cache_frequency = 2  # GHz
cache_core = 16 # bit
cache_cycle = crossbar_cycle * crossbar_frequency / cache_frequency  # cycle

# specs for nonlinear function
num_decomposed_round = 20


# indicator for the utilization of the crossbar
ult_memory_read_time = 0
ult_memory_write_time = 0
ult_crossbar_time = 0
ult_crossbar_percentage = 0
ult_ADC_time = 0
ult_total_time = 0


# print the details of the simulation
print_details = False





# simulate nonlinear activation
def nonlinear(in_features, batch_size):
    # the nonlinear function has two stage:
    # 1. compute the muliplier, e.g. 1x 2x 3x ... 20x
    # 2. compute fourier transform, e.g. sin(1x) sin(2x) sin(3x) ... sin(20x)
    # the runtime is the sum of the two stages

    # compute the round for the multiplier
    global ult_memory_read_time
    global ult_memory_write_time
    global ult_crossbar_time
    global ult_ADC_time
    global ult_total_time

    rounds = math.ceil(in_features * batch_size / crossbar_size)
    if print_details:
        print('The iteration for computing over input feature is: ', rounds)
    cycle = 0
    for i in range(rounds):
        # step 1: compute the multiplier

        #### memory read
        if i == 0:
            cycle_memory_lookup = memory_address_look_cycle
        else:
            cycle_memory_lookup = 0
        cycle_load_input = math.ceil(crossbar_size / memory_bandwidth_per_cycle)
        cycle_read = cycle_memory_lookup + cycle_load_input
        ult_memory_read_time += cycle_read

        #### crossbar stage
        computation_cycle = 10
        cycle_crossbar = computation_cycle
        ult_crossbar_time += cycle_crossbar
        #### readout stage
        # readout the result from the crossbar
        cycle_readout = readout_cycle * math.ceil(num_decomposed_round / reader_batch)
        ult_ADC_time += cycle_readout


        # step 2: compute the fourier transform
        #### crossbar stage
        computation_cycle = num_decomposed_round
        cycle_crossbar += computation_cycle
        ult_crossbar_time += computation_cycle
        #### readout stage
        # readout the result from the crossbar
        cycle_readout += readout_cycle
        ult_ADC_time += cycle_readout

        #### write the result back to the memory
        cycle_write_output = math.ceil(crossbar_size * num_decomposed_round / memory_bandwidth_per_cycle)
        ult_memory_write_time += cycle_write_output
        #### the runtime is the maximum of the three stages
        cycle += max(cycle_crossbar + cycle_readout, cycle_write_output + cycle_read)
    return cycle





# simulate addition operation
def matrix_addition(in_features_width, in_features_height, channel, batch_size):
    # simulate two matrix M1+M2
    # M1 and M2 are both in_features_width*in_features_height*batch_size
    # compute the iteration for sum up
    global ult_memory_read_time
    global ult_memory_write_time
    global ult_crossbar_time
    global ult_ADC_time
    global ult_total_time

    cross_bar_adder = crossbar_size * 2 - 2
    rounds = math.ceil(in_features_width * in_features_height * batch_size * channel / cross_bar_adder)

    if print_details:
        print('[add] The iteration for computing over input feature is: ', rounds)
    cycle = 0
    for j in range(rounds // reader_batch):
        ####memory read stage
        # load the input data into the input cache
        if j == 0:
            cycle_load_input = memory_address_look_cycle
        else:
            cycle_load_input = 0
        cycle_load_input += math.ceil(crossbar_size * 2 * reader_batch / memory_bandwidth_per_cycle)
        ult_memory_read_time += cycle_load_input
        cycle_crossbar = 0
        for i in range(reader_batch):
            # Each round process crossbar_size rows of input
            ####crossbar stage
            # crossbar computation & memory read
            computation_cycle = crossbar_element_wise_cycle
            cycle_crossbar += computation_cycle

        #### readout stage
        # readout the result from the crossbar
        cycle_readout = readout_cycle
        ult_ADC_time += cycle_readout

        #### write the result back to the memory
        if j == 0:
            cycle_write_output = memory_address_look_cycle
        else:
            cycle_write_output = 0
        cycle_write_output += math.ceil(crossbar_size * reader_batch / memory_bandwidth_per_cycle)
        ult_memory_write_time += cycle_write_output

        #### the runtime is the maximum of the three stages
        cycle += max(cycle_crossbar + cycle_readout, cycle_write_output + cycle_load_input)
    if print_details:
        print('[add] The total cycle for the crossbar stage is: ', cycle)
    return cycle





# simulate a fully connected layer
def linear(in_features, out_features, batch_size):
    ####crossbar stage
    # maximum number of input features that can be processed in one pulse
    global ult_memory_read_time
    global ult_memory_write_time
    global ult_crossbar_time
    global ult_ADC_time
    global ult_total_time

    if max_pulse < in_features:
        # if the input feature is larger than the maximum pulse
        in_features_per_round = max_pulse
        rounds = math.ceil(in_features / max_pulse)
    else:
        # if the input feature is smaller than the maximum pulse
        in_features_per_round = in_features
        rounds = 1
    # compute the iteration for input feature
    if batch_size > crossbar_size:
        iteration_feature = math.ceil(batch_size / crossbar_size)
        batch_per_round = crossbar_size
    else:
        iteration_feature = 1
        batch_per_round = batch_size
    if print_details:
        print('The iteration for computing over input feature is: ', iteration_feature)

    # compute the iteration for output feature
    if out_features > crossbar_size:
        iteration_out_feature = math.ceil(out_features / crossbar_size)
        out_channel_per_round = crossbar_size
    else:
        iteration_out_feature = 1
        out_channel_per_round = out_features
    if print_details:
        print('The iteration for computing over output feature is: ', iteration_out_feature)

    # calculate the actual cache size
    weight_cache_size_actual = in_features_per_round * batch_per_round
    input_cache_size_actual = crossbar_size * in_features_per_round
    output_cache_size_actual = crossbar_size * out_channel_per_round

    # transform the cache size by the precision
    weight_cache_size_actual = weight_cache_size_actual * input_bit_precision / 8
    input_cache_size_actual = input_cache_size_actual * input_bit_precision / 8
    output_cache_size_actual = output_cache_size_actual * readout_bit_precision / 8
    if print_details:
        print("[Linear] Required input cache size (MB): ", input_cache_size_actual / 1024 / 1024)  # convert to MB
        print("[Linear] Required weight cache size (MB): ", weight_cache_size_actual / 1024 / 1024)  # convert to MB
        print("[Linear] Required output cache size (MB): ", output_cache_size_actual / 1024 / 1024)  # convert to MB

    # -----start simulation for crossbar stage
    # initial the timer to accumulate the runtime
    cycle = 0

    for i in range(iteration_out_feature):
        # totally iteration output feature rounds
        # 1. load the kth batch and ith output weights into weight cache
        cycle_memory_lookup = memory_address_look_cycle
        cycle_load_weight = math.ceil(weight_cache_size_actual / memory_bandwidth_per_cycle)
        ult_memory_read_time += cycle_load_weight
        cycle += cycle_load_weight + cycle_memory_lookup
        for j in range(iteration_feature):
            # Each round process crossbar_size rows of input
            for k in range(rounds):
                # Each round process max_pulse input features
                ####memory read stage
                # load the input data into the input cache
                if k == 0:
                    cycle_read_input = memory_address_look_cycle
                    cycle_read_input += math.ceil(input_cache_size_actual / memory_bandwidth_per_cycle)
                    ult_memory_read_time += math.ceil(input_cache_size_actual / memory_bandwidth_per_cycle)
                else:
                    cycle_read_input += math.ceil(input_cache_size_actual / memory_bandwidth_per_cycle)
                    ult_memory_read_time += math.ceil(input_cache_size_actual / memory_bandwidth_per_cycle)

                #### crossbar stage
                # crossbar computation & memory read
                computation_cycle = in_features_per_round
                cycle_crossbar = computation_cycle
                ult_crossbar_time += computation_cycle

                #### readout stage
                # readout the result from the crossbar
                cycle_readout = readout_cycle * out_channel_per_round / reader_batch
                ult_ADC_time += cycle_readout

                #### write the result back to the memory
                cycle_write_output = memory_address_look_cycle
                cycle_write_output += math.ceil(output_cache_size_actual / memory_bandwidth_per_cycle)
                ult_memory_write_time += math.ceil(output_cache_size_actual / memory_bandwidth_per_cycle)

                #### the runtime is the maximum of the three stages
                cycle += max(cycle_crossbar + cycle_readout, cycle_read_input + cycle_write_output)

            if rounds > 1:
                # sum up the result for all input features
                # read the output data from the memory
                cycle_read_output_sum = memory_address_look_cycle
                cycle_read_output_sum += math.ceil(output_cache_size_actual * rounds / memory_bandwidth_per_cycle)
                ult_memory_read_time += math.ceil(output_cache_size_actual * rounds / memory_bandwidth_per_cycle)

                # compute the rounds for sum up the output
                crossbar_size_adder = crossbar_size * 2 - 2
                sum_up_round = math.ceil(output_cache_size_actual / crossbar_size_adder / rounds)
                cycle_sum_up = 0
                for k in range(sum_up_round):
                    # sum up the output data
                    cycle_compute_sum = rounds*crossbar_element_wise_cycle
                    ult_crossbar_time += cycle_compute_sum
                    cycle_readout = readout_cycle
                    ult_ADC_time += cycle_readout
                    cycle_sum_up += cycle_compute_sum + cycle_readout


                cycle_write_sum = memory_address_look_cycle
                cycle_write_sum += math.ceil(output_cache_size_actual / memory_bandwidth_per_cycle)
                ult_memory_write_time += math.ceil(output_cache_size_actual / memory_bandwidth_per_cycle)

                cycle += cycle_sum_up + cycle_read_output_sum + cycle_write_sum
    if print_details:
        print('[linear] The total cycle for the crossbar stage is: ', cycle)
        print('[linear] The cycle per round for the crossbar is: ', cycle_crossbar)
        print('[linear] The cycle per round for the readout is: ', cycle_readout)

    return cycle


# simulate a convolutional layer
def conv2d(in_channels, out_channel, feature_size, kernel_size, stride, padding, batch_size):
    # convolution has 2 pipelined stages: crossbar for multiplication and adder for accumulation
    # the runtime is the maximum of the two stages
    global ult_memory_read_time
    global ult_memory_write_time
    global ult_crossbar_time
    global ult_ADC_time
    global ult_total_time

    if print_details:
        print('[conv] The input feature size is: ', feature_size,'in_channels: ', in_channels, 'out_channel: ', out_channel)
    ####crossbar stage
    feature_size = feature_size / stride

    # maximum number of input channels that can be processed in one crossbar cycle
    max_in_channels_per_round = math.ceil(max_pulse / kernel_size / kernel_size)  # max_pulse = 128*9
    if in_channels < max_in_channels_per_round:
        in_channels_per_round = in_channels * kernel_size * kernel_size
        in_channel_round = 1
    else:
        in_channels_per_round = max_in_channels_per_round * kernel_size * kernel_size
        in_channel_round = math.ceil(in_channels / max_in_channels_per_round)
    if print_details:
        print('[conv] The number of the rounds for split the input channel: ', in_channel_round)

    # compute the iteration for input channel
    if feature_size*feature_size > crossbar_size:
        rows_per_round = math.floor(
            crossbar_size / feature_size)  # how many rows of input can be processed in one crossbar cycle
        if rows_per_round == 0: # if the input feature map is larger than the crossbar size
            iteration_feature = math.ceil(feature_size / crossbar_size) * feature_size  # how many iterations for input
            memory_per_round = (rows_per_round + padding) * (crossbar_size + padding) * in_channels_per_round
        else: # if the input feature map is smaller than the crossbar size
            iteration_feature = math.ceil(feature_size / (math.floor(crossbar_size / feature_size))) # how many iterations for input
            memory_per_round = (rows_per_round + padding) * (feature_size + padding) * in_channels_per_round
    else:
        iteration_feature = 1
        memory_per_round = (feature_size + padding) * (feature_size + padding) * in_channels_per_round
    if print_details:
        print('[conv] The iteration for computing over input feature map is: ', iteration_feature)

    # compute the iteration for out_channel
    if feature_size * feature_size > crossbar_size:
        iteration_out_channel = math.ceil(out_channel / crossbar_size)
    else:
        iteration_out_channel = 1
    if print_details:
        print('[conv] The iteration for computing over output channel is: ', iteration_out_channel)

    # calculate the actual cache size
    weight_cache_size_actual = in_channels_per_round * crossbar_size
    input_cache_size_actual = memory_per_round
    output_cache_size_actual = crossbar_size * max(out_channel, crossbar_size) * iteration_out_channel

    # transform the cache size by the precision
    weight_cache_size_actual = weight_cache_size_actual * input_bit_precision / 8
    input_cache_size_actual = input_cache_size_actual * input_bit_precision / 8
    output_cache_size_actual = output_cache_size_actual * readout_bit_precision / 8
    if print_details:
        print("[conv] Required input cache size (MB): ", input_cache_size_actual / 1024 / 1024)  # convert to MB
        print("[conv] Required weight cache size (MB): ", weight_cache_size_actual / 1024 / 1024)  # convert to MB
        print("[conv] Required output cache size (MB): ", output_cache_size_actual / 1024 / 1024)  # convert to MB

    # -----start simulation for crossbar stage
    # initial the timer to accumulate the runtime
    cycle = 0

    for i in range(iteration_out_channel):
        # totally iteration output channel rounds
        for k in range(in_channel_round):
            # Each round process crossbar_size rows of input
            ####memory read stage
            # 1. load the kth in_channel and ith out_channel weights into weight cache
            cycle_memory_lookup = memory_address_look_cycle
            cycle_load_weight = math.ceil(weight_cache_size_actual / memory_bandwidth_per_cycle)  # load 225B/cycle
            cycle += cycle_load_weight + cycle_memory_lookup
            ult_memory_read_time += cycle_load_weight * batch_size
            for j in range(int(iteration_feature)):
                # Each round process crossbar_size rows of input
                ####memory read stage
                # load the input data into the input cache
                if j == 0:
                    cycle_read_input = memory_address_look_cycle
                else:
                    cycle_read_input = 0

                cycle_read_input += math.ceil(memory_per_round / memory_bandwidth_per_cycle)  # load 225B/cycle
                ult_memory_read_time += math.ceil(memory_per_round / memory_bandwidth_per_cycle) * batch_size
                cycle_crossbar = 0
                for m in range(kernel_size * kernel_size):
                    # store the input data into the crossbar buffer
                    if m == 0:
                        cycle_store_input = cache_cycle  # required cycles to store the input data
                    else:
                        cycle_store_input = 0

                    ####crossbar stage
                    # crossbar computation & memory read
                    computation_cycle = in_channels_per_round / kernel_size / kernel_size
                    cycle_crossbar += max(computation_cycle,
                                         cycle_store_input)  # the runtime is the max of the two processes
                    ult_crossbar_time += computation_cycle  * batch_size
                ####readout stage
                # readout the result from the crossbar
                cycle_readout = readout_cycle * crossbar_size / reader_batch
                ult_ADC_time += cycle_readout  * batch_size

                #### write the result back to the memory
                if j == 0:
                    cycle_write_output = memory_address_look_cycle
                else:
                    cycle_write_output = 0
                cycle_write_output += math.ceil(crossbar_size * max(out_channel, crossbar_size) / memory_bandwidth_per_cycle)
                ult_memory_write_time += math.ceil(crossbar_size * max(out_channel, crossbar_size) / memory_bandwidth_per_cycle) * batch_size

                #### the runtime is the maximum of the three stages
                cycle += max(cycle_crossbar + cycle_readout, cycle_read_input + cycle_write_output)

        if in_channel_round > 1:
            #### sum up the result for all input channel
            # read the output data from the memory
            cycle_read_output_sum = memory_address_look_cycle
            cycle_read_output_sum += math.ceil(output_cache_size_actual / memory_bandwidth_per_cycle)
            ult_memory_read_time += math.ceil(output_cache_size_actual / memory_bandwidth_per_cycle)  * batch_size

            # compute the rounds for sum up the output
            crossbar_adder = crossbar_size * 2 - 2
            sum_up_round = math.ceil(output_cache_size_actual / crossbar_adder / in_channel_round)
            cycle_sum_up = 0
            for j in range(sum_up_round):
                # sum up the output data
                cycle_store_output = cache_cycle # required cycles to store the output data to the cache

                cycle_compute_sum = in_channel_round * crossbar_element_wise_cycle
                ult_crossbar_time += (cycle_compute_sum + cycle_store_output) * batch_size

                cycle_readout = readout_cycle  # assume only one cycle to readout the result
                ult_ADC_time += cycle_readout * batch_size

                cycle_sum_up += max(max(cycle_store_output, cycle_compute_sum),cycle_readout) # the runtime is the max of the three processes

            cycle_write_sum = memory_address_look_cycle
            cycle_write_sum += math.ceil(output_cache_size_actual/in_channel_round / memory_bandwidth_per_cycle)
            ult_memory_write_time += math.ceil(output_cache_size_actual/in_channel_round / memory_bandwidth_per_cycle) * batch_size
            if print_details:
                print('[conv] The total cycle for the sum up is: ', cycle_write_sum)
            cycle += cycle_sum_up + cycle_read_output_sum + cycle_write_sum

    if print_details:
        print('[conv] The total cycle for the crossbar stage is: ', cycle)
        print('[conv] The cycle per round for the crossbar is: ', cycle_crossbar)
        print('[conv] The cycle per round for the readout is: ', cycle_readout)
    return cycle * batch_size


def memory_read(in_features, batch_size):
    # the memory read has two stage:
    # 1. compute the address lookup
    # 2. compute the data read
    global ult_memory_read_time
    global ult_memory_write_time
    global ult_crossbar_time
    global ult_ADC_time
    global ult_total_time

    rounds = math.ceil(in_features * batch_size / crossbar_size)
    if print_details:
        print('The iteration for computing over input feature is: ', rounds)
    cycle = 0
    for i in range(rounds):
        # step 1: compute the address lookup
        cycle_memory_lookup = memory_address_look_cycle
        cycle_load_input = math.ceil(crossbar_size / memory_bandwidth_per_cycle)
        cycle_read = cycle_memory_lookup + cycle_load_input
        ult_memory_read_time += cycle_read

        # step 2: compute the data read
        cycle += cycle_read

    return cycle

def memory_write(in_features, batch_size):
    # the memory write has two stage:
    # 1. compute the address lookup
    # 2. compute the data write
    global ult_memory_read_time
    global ult_memory_write_time
    global ult_crossbar_time
    global ult_ADC_time
    global ult_total_time

    rounds = math.ceil(in_features * batch_size / crossbar_size)
    if print_details:
        print('The iteration for computing over input feature is: ', rounds)
    cycle = 0
    for i in range(rounds):
        # step 1: compute the address lookup
        cycle_memory_lookup = memory_address_look_cycle
        cycle_load_input = math.ceil(crossbar_size / memory_bandwidth_per_cycle)
        cycle_read = cycle_memory_lookup + cycle_load_input
        ult_memory_write_time += cycle_read

        # step 2: compute the data write
        cycle += cycle_read

    return cycle

# simulate attention layer
def llm_attention(token_features, q_features, kv_features, in_seq_length, seq_length, num_heads, MLP_feature, batch_size):
    # the attention layer is a fully connected layer
    # with a nonlinear activation function
    cycle = 0
    # first step: compute the QKV
    cycle_q = linear(token_features, token_features, batch_size*in_seq_length)
    cycle_k = linear(token_features, kv_features, batch_size*in_seq_length)
    cycle_v = linear(token_features, kv_features, batch_size*in_seq_length)

    cycle += cycle_q + cycle_k + cycle_v

    # second step: append to the cache sequence
    cycle_read_cache = memory_read(kv_features * seq_length, batch_size)
    cycle_write_cache = memory_write(kv_features * (in_seq_length + seq_length), batch_size)

    cycle += cycle_read_cache + cycle_write_cache

    # third step: padding the KV to the output sequence length

    n_rep = q_features / kv_features
    cycle_fill_k = memory_write(kv_features * (seq_length + in_seq_length) * n_rep, batch_size)
    cycle_fill_v = memory_read(kv_features * (seq_length + in_seq_length) * n_rep, batch_size)

    cycle += cycle_fill_k + cycle_fill_v

    # fourth step: compute the attention
    head_features = q_features / num_heads
    cycle_score = 0
    cycle_softmax = 0

    cycle_score += linear(seq_length, head_features,batch_size)*num_heads

    cycle_softmax += nonlinear(seq_length*batch_size, seq_length)*num_heads

    cycle_score_v = 0

    cycle_score_v += linear(seq_length, head_features, batch_size)*num_heads

    cycle_out_formatting = memory_write(seq_length * num_heads * head_features, batch_size)
    cycle_attention = cycle_score + cycle_score_v + cycle_softmax + cycle_out_formatting


    # fifth step: compute the MLP layer
    cycle_w1 = 0
    cycle_w2 = 0
    cycle_silu = 0
    cycle_w3 = 0
    for i in range(in_seq_length):
        cycle_w1 += linear(q_features, MLP_feature, batch_size)
        cycle_w2 += linear(q_features, MLP_feature, batch_size)

        cycle_silu += nonlinear(MLP_feature, batch_size)
        cycle_w3 += linear(MLP_feature, q_features, batch_size)

    cycle += cycle_attention + cycle_w1 + cycle_w2 + cycle_silu + cycle_w3
    return cycle



# print the utilization for each component
def print_utilization(total_time=0, batch_size=32):
    global ult_memory_read_time
    global ult_memory_write_time
    global ult_crossbar_time
    global ult_ADC_time
    global ult_total_time
    global print_details

    ult_total_time = total_time
    if print_details:
        print('The utilization for the memory read is: ', ult_memory_read_time / ult_total_time)
        print('The utilization for the memory write is: ', ult_memory_write_time / ult_total_time)
        print('The utilization for the crossbar is: ', ult_crossbar_time / ult_total_time)
        print('The utilization for the ADC is: ', ult_ADC_time / ult_total_time)
        print('The total runtime is: ', ult_total_time)

    # reset the timer
    ult_memory_read_time = 0
    ult_memory_write_time = 0
    ult_crossbar_time = 0
    ult_ADC_time = 0
    ult_total_time = 0










def main():
    args = parser.parse_args()
    global print_details

    print_details = args.print_utilization

    eval_function = args.evaluate

    batch_size = args.batch_size

    print('starting the simulation for ', eval_function, ', the batch size is: ', batch_size)

    if eval_function == 'basic_ops':
        # test conv2d
        print('conv2d test:')
        cycle = conv2d(64, 64, 224, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)
        cycle = conv2d(256, 256, 224, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)
        cycle = conv2d(512, 512, 56, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)

        print('conv2d test:')
        cycle = conv2d(64, 64, 64, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)
        cycle = conv2d(256, 256, 64, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)
        cycle = conv2d(512, 512, 64, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)

        print('conv2d test:')
        cycle = conv2d(64, 64, 32, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)
        cycle = conv2d(256, 256, 32, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)
        cycle = conv2d(512, 512, 32, 3, 1, 1, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, batch_size)

        # test linear
        print('linear test:')
        cycle = linear(512, 512, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, 1)
        cycle = linear(1024, 1024, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, 1)
        cycle = linear(2048, 2048, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, 1)
        cycle = linear(4096, 4096, batch_size)
        print(cycle / 1e6 / crossbar_frequency / crossbar_cycle)
        print_utilization(cycle, 1)

        # test add
        #print(matrix_addition(512, 512, 1, batch_size) / 1e6 / crossbar_frequency / crossbar_cycle)
        #print(matrix_addition(1024, 1024, 1, batch_size) / 1e6 / crossbar_frequency / crossbar_cycle)
        #print(matrix_addition(2048, 2048, 1, batch_size) / 1e6 / crossbar_frequency / crossbar_cycle)
        #print(matrix_addition(4096, 4096, 1, batch_size) / 1e6 / crossbar_frequency / crossbar_cycle)


    # test conv network
    if eval_function == 'conv_network':
        for dataset in ['cifar100', 'imagenet', 'cifar10']:
            print('The dataset is: ', dataset)
            for model in ['resnet18', 'resnet50', 'resnet101', 'mobilenet_v2', 'mobilenet_v3_large', 'VGG16', 'VGG11', 'VGG19']:
                model_eval(model, dataset, batch_size)

    # test attention network
    if eval_function == 'attention_network':
        ## llama 3.1-8B

        token_class = 128256
        token_features = 4096
        q_features = 4096
        kv_features = 1024
        start_pos = 8
        seq_length = 128
        num_heads = 32
        MLP_feature = 14336
        #batch_size = 4

        layers = 32

        # token embedding
        cycle = 0
        cycle += linear(token_class, token_features, batch_size)

        last_pos = 0
        # attention layers
        for cur_pos in range(start_pos, seq_length):

            cycle += llm_attention(token_features, q_features, kv_features, cur_pos-last_pos, seq_length, num_heads, MLP_feature, batch_size)*layers

            print(cycle)
            last_pos = cur_pos

        # output embedding
        cycle += linear(token_features, token_class, batch_size)

        # print the total cycle
        print('The total cycle for llama 3.1-8B is: ', cycle)
        print('The total time for llama 3.1-8B is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')

        ## vision transformer / B - 16
        image_size = 384
        patch_size = 16
        num_patches = (image_size // patch_size) ** 2
        token_class = 768
        token_features = 768
        q_features = 768
        kv_features = 768

        output_class = 1000
        num_heads = 12

        #batch_size = 32

        layers = 12
        MLP_feature = 3072



        cycle = 0
        # patch embedding
        cycle += memory_read(image_size * image_size * 3, batch_size)
        cycle += linear(3 * image_size * image_size, token_features, batch_size)
        cycle += memory_write(token_features * num_patches, batch_size)

        # attention layers
        for i in range(layers):
            cycle += llm_attention(token_features, q_features, kv_features, num_patches, num_patches, num_heads, MLP_feature, batch_size)

        # output embedding
        cycle += memory_read(token_features * num_patches, batch_size)
        cycle += linear(token_features, output_class, batch_size)
        cycle += memory_write(output_class, batch_size)

        # print the total cycle
        print('The total cycle for vision transformer is: ', cycle)
        print('The total time for vision transformer is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')


        ## Bert
        token_class = 30522
        token_features = 768

        q_features = 768
        kv_features = 768
        start_pos = 8
        seq_length = 128
        num_heads = 12
        MLP_feature = 3072
        #batch_size = 128

        layers = 12

        # token embedding
        cycle = 0
        cycle += linear(token_class, token_features, batch_size)
        # position embedding
        cycle += linear(512, token_features, batch_size)

        # attention layers

        cycle += llm_attention(token_features, q_features, kv_features, seq_length, seq_length, num_heads, MLP_feature, batch_size)*layers

        # output embedding
        cycle += linear(token_features, token_features, batch_size)

        cycle += memory_write(token_features, batch_size)
        # print the total cycle
        print('The total cycle for Bert is: ', cycle)
        print('The total time for Bert is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')


def model_eval(model, dataset, batch_size):

    if dataset == 'cifar100':
        input_size = 32
        output_size = 100
    elif dataset == 'imagenet':
        input_size = 224
        output_size = 1000
    elif dataset == 'cifar10':
        input_size = 32
        output_size = 10


    if model == 'resnet18':
        cycle = 0
        cycle_add = 0
        # simulate for a resnet18
        # first layer (conv1)
        cycle = cycle + conv2d(3, 64, input_size, 7, 2, 1, batch_size)
        #print('first layer: ', cycle)
        # second layer (layer1)
        feature_size = input_size // 2
        for i in range(2):
            cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 64, batch_size)
            cycle_add = cycle_add + matrix_addition(feature_size, feature_size, 64, batch_size)
        # third layer (layer2)
        for i in range(2):
            if i == 0:
                cycle = cycle + conv2d(64, 128, feature_size, 3, 2, 1, batch_size)
            else:
                cycle = cycle + conv2d(128, 128, feature_size / 2, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(128, 128, feature_size / 2, 3, 1, 1, batch_size)
            cycle = cycle + matrix_addition(feature_size / 2, feature_size / 2, 128, batch_size)
            cycle_add = cycle_add + matrix_addition(feature_size / 2, feature_size / 2, 128, batch_size)
        # forth layer (layer3)
        feature_size = feature_size // 2
        for i in range(2):
            if i == 0:
                cycle = cycle + conv2d(128, 256, feature_size, 3, 2, 1, batch_size)
            else:
                cycle = cycle + conv2d(256, 256, feature_size / 2, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(256, 256, feature_size / 2, 3, 1, 1, batch_size)
            cycle = cycle + matrix_addition(feature_size / 2, feature_size / 2, 256, batch_size)
            cycle_add = cycle_add + matrix_addition(feature_size / 2, feature_size / 2, 256, batch_size)
        # fifth layer (layer4)
        feature_size = feature_size // 2
        for i in range(2):
            if i == 0:
                cycle = cycle + conv2d(256, 512, feature_size, 3, 2, 1, batch_size)
            else:
                cycle = cycle + conv2d(512, 512, feature_size / 2, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(512, 512, feature_size / 2, 3, 1, 1, batch_size)
            cycle = cycle + matrix_addition(feature_size / 2, feature_size / 2, 512, batch_size)
            cycle_add = cycle_add + matrix_addition(feature_size / 2, feature_size / 2, 512, batch_size)
        # seventh layer (fc)
        cycle = cycle + linear(512, output_size, batch_size)

        cycle = cycle + nonlinear(output_size, batch_size)
        print_utilization(cycle, batch_size)

        print('The total cycle for the resnet18 is: ', cycle)
        print('The total time for the resnet18 is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')
    elif model == 'resnet50':
        cycle = 0
        # simulate for a resnet50
        # first layer (conv1)

        cycle = cycle + conv2d(3, 64, input_size, 7, 2, 3, batch_size)
        #print('first layer: ', cycle)
        # second layer (layer1)
        feature_size = input_size // 2
        for i in range(3):
            if i == 0:
                cycle = cycle + conv2d(64, 64, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(64, 64, feature_size, 3, 2, 1, batch_size)
                cycle = cycle + conv2d(64, 256, feature_size, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(256, 64, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(64, 256, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 256, batch_size)
        # third layer (layer2)
        for i in range(4):
            if i == 0:
                cycle = cycle + conv2d(256, 128, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(128, 128, feature_size, 3, 2, 1, batch_size)
                cycle = cycle + conv2d(128, 512, feature_size/2, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(512, 128, feature_size/2, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(128, 128, feature_size/2, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(128, 512, feature_size/2, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size/2, feature_size/2, 512, batch_size)
        # forth layer (layer3)
        feature_size = feature_size // 2
        for i in range(6):
            if i == 0:
                cycle = cycle + conv2d(512, 256, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(256, 256, feature_size, 3, 2, 1, batch_size)
                cycle = cycle + conv2d(256, 1024, feature_size/2, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(1024, 256, feature_size/2, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(256, 256, feature_size/2, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(256, 1024, feature_size/2, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size/2, feature_size/2, 1024, batch_size)
        # fifth layer (layer4)
        feature_size = feature_size // 2
        for i in range(3):
            if i == 0:
                cycle = cycle + conv2d(1024, 512, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(512, 512, feature_size, 3, 2, 1, batch_size)
                cycle = cycle + conv2d(512, 2048, feature_size/2, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(2048, 512, feature_size/2, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(512, 512, feature_size/2, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(512, 2048, feature_size/2, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size / 2, feature_size / 2, 2048, batch_size)
        # seventh layer (fc)
        cycle = cycle + linear(2048, output_size, batch_size)
        cycle = cycle + nonlinear(output_size, batch_size)

        print_utilization(cycle, batch_size)
        print('The total cycle for the resnet50 is: ', cycle)
        print('The total time for the resnet50 is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')

    elif model == 'resnet101':
        cycle = 0
        # simulate for a resnet101
        # first layer (conv1)
        cycle = cycle + conv2d(3, 64, input_size, 7, 2, 3, batch_size)
        #print('first layer: ', cycle)
        # second layer (layer1)
        feature_size = input_size // 2
        for i in range(3):
            if i == 0:
                cycle = cycle + conv2d(64, 64, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(64, 256, feature_size, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(256, 64, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(64, 256, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(112, 112, 256, batch_size)
        # third layer (layer2)
        for i in range(4):
            if i == 0:
                cycle = cycle + conv2d(256, 128, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(128, 128, feature_size, 3, 2, 1, batch_size)
                cycle = cycle + conv2d(128, 512, feature_size / 2, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(512, 128, feature_size / 2, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(128, 128, feature_size / 2, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(128, 512, feature_size / 2, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(56, 56, 512, batch_size)
        # forth layer (layer3)
        feature_size = feature_size // 2
        for i in range(23):
            if i == 0:
                cycle = cycle + conv2d(512, 256, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(256, 256, feature_size, 3, 2, 1, batch_size)
                cycle = cycle + conv2d(256, 1024, feature_size / 2, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(1024, 256, feature_size / 2, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(256, 256, feature_size / 2, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(256, 1024, feature_size / 2, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(28, 28, 1024, batch_size)
        # fifth layer (layer4)
        feature_size = feature_size // 2
        for i in range(3):
            if i == 0:
                cycle = cycle + conv2d(1024, 512, feature_size, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(512, 512, feature_size, 3, 2, 1, batch_size)
                cycle = cycle + conv2d(512, 2048, feature_size / 2, 1, 1, 0, batch_size)
            else:
                cycle = cycle + conv2d(2048, 512, feature_size / 2, 1, 1, 0, batch_size)
                cycle = cycle + conv2d(512, 512, feature_size / 2, 3, 1, 1, batch_size)
                cycle = cycle + conv2d(512, 2048, feature_size / 2, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(14, 14, 2048, batch_size)
        # seventh layer (fc)
        cycle = cycle + linear(2048, output_size, batch_size)
        cycle = cycle + nonlinear(output_size, batch_size)
        print_utilization(cycle, batch_size)
        print('The total cycle for the resnet101 is: ', cycle)
        print('The total time for the resnet101 is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')
    elif model == 'mobilenet_v2':
        cycle = 0
        # simulate for a mobilenet_v2
        # first layer (conv1)
        cycle = cycle + conv2d(3, 32, input_size, 3, 2, 1, batch_size)
        cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        #Bottole neck layer 1
        feature_size = input_size // 2
        for i in range(cfgs[0][2]):
            cycle = cycle + conv2d(32, 32, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + conv2d(32, 32, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(32, 128, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(112, 112,128, batch_size)
        #Bottole neck layer 2
        feature_size = feature_size // 2
        for i in range(cfgs[1][2]):
            cycle = cycle + conv2d(128, 32, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + conv2d(32, 32, feature_size, 3, 2, 1, batch_size)
            cycle = cycle + conv2d(32, 128, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 128, batch_size)
        #Bottole neck layer 3
        for i in range(cfgs[2][2]):
            cycle = cycle + conv2d(128, 32, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + conv2d(32, 32, feature_size, 3, 2, 1, batch_size)
            cycle = cycle + conv2d(32, 128, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 128, batch_size)
        #Bottole neck layer 4
        feature_size = feature_size // 2
        for i in range(cfgs[3][2]):
            cycle = cycle + conv2d(128, 64, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + conv2d(64, 64, feature_size, 3, 2, 1, batch_size)
            cycle = cycle + conv2d(64, 256, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 256, batch_size)
        #Bottole neck layer 5
        for i in range(cfgs[4][2]):
            cycle = cycle + conv2d(256, 64, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(64, 256, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 256, batch_size)
        #Bottole neck layer 6
        feature_size = feature_size // 2
        for i in range(cfgs[5][2]):
            cycle = cycle + conv2d(256, 128, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + conv2d(128, 128, feature_size, 3, 2, 1, batch_size)
            cycle = cycle + conv2d(128, 512, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 512, batch_size)
        #Bottole neck layer 7
        for i in range(cfgs[6][2]):
            cycle = cycle + conv2d(512, 128, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + conv2d(128, 128, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(128, 512, feature_size, 1, 1, 0, batch_size)
            cycle = cycle + matrix_addition(feature_size, feature_size, 512, batch_size)
        # eighth layer (fc)
        cycle = cycle + linear(1028, output_size, batch_size)
        cycle = cycle + nonlinear(output_size, batch_size)
        print_utilization(cycle, batch_size)
        print('The total cycle for the mobilenet_v2 is: ', cycle)
        print('The total time for the mobilenet_v2 is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')
    elif model == 'VGG11':
        cycle = 0
        # simulate for a VGG11
        # first layer (conv1)
        cycle = cycle + conv2d(3, 64, input_size, 3, 1, 1, batch_size)
        #print('first layer: ', cycle)
        # second layer (layer1)
        feature_size = input_size // 2
        for i in range(1):
            cycle = cycle + conv2d(64, 128, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(128, 128, feature_size, 3, 1, 1, batch_size)

        # third layer (layer2)
        feature_size = feature_size // 2
        for i in range(2):

            cycle = cycle + conv2d(128, 256, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(256, 256, feature_size, 3, 1, 1, batch_size)

        # forth layer (layer3)
        feature_size = feature_size // 2
        for i in range(2):
            cycle = cycle + conv2d(256, 512, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(512, 512, feature_size, 3, 1, 1, batch_size)

        # fifth layer (layer4)
        feature_size = feature_size // 2
        for i in range(2):
            cycle = cycle + conv2d(512, 512, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(512, 512, feature_size, 3, 1, 1, batch_size)

        # seventh layer (fc)
        feature_size = feature_size // 2
        cycle = cycle + linear(512*feature_size*feature_size, 4096, batch_size)
        cycle = cycle + linear(4096, 4096, batch_size)
        cycle = cycle + linear(4096, output_size, batch_size)
        cycle = cycle + nonlinear(output_size, batch_size)
        print_utilization(cycle, batch_size)
        print('The total cycle for the VGG11 is: ', cycle)
        print('The total time for the VGG11 is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')
    elif model == 'VGG16':
        cycle = 0
        # simulate for a VGG16
        # first layer (conv1)
        cycle = cycle + conv2d(3, 64, input_size, 3, 1, 1, batch_size)
        #print('first layer: ', cycle)
        feature_size = input_size // 2
        # second layer (layer1)
        for i in range(2):
            cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
        # third layer (layer2)
        feature_size = feature_size // 2
        for i in range(2):
            cycle = cycle + conv2d(64, 128, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(128, 128, feature_size, 3, 1, 1, batch_size)
        # forth layer (layer3)
        feature_size = feature_size // 2
        for i in range(3):
            cycle = cycle + conv2d(128, 256, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(256, 256, feature_size, 3, 1, 1, batch_size)
        # fifth layer (layer4)
        feature_size = feature_size // 2
        for i in range(3):
            cycle = cycle + conv2d(256, 512, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(512, 512, feature_size, 3, 1, 1, batch_size)

        # seventh layer (fc)
        feature_size = feature_size // 2
        cycle = cycle + linear(512 * feature_size * feature_size, 4096, batch_size)
        cycle = cycle + linear(4096, 4096, batch_size)
        cycle = cycle + linear(4096, output_size, batch_size)
        cycle = cycle + nonlinear(output_size, batch_size)
        print_utilization(cycle, batch_size)
        print('The total cycle for the VGG16 is: ', cycle)
        print('The total time for the VGG16 is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')
    elif model == 'VGG19':
        cycle = 0
        # simulate for a VGG19
        # first layer (conv1)
        cycle = cycle + conv2d(3, 64, input_size, 3, 1, 1, batch_size)
       # print('first layer: ', cycle)
        # second layer (layer1)
        feature_size = input_size // 2
        for i in range(2):
            cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(64, 64, feature_size, 3, 1, 1, batch_size)
        # third layer (layer2)
        feature_size = feature_size // 2
        for i in range(4):
            cycle = cycle + conv2d(64, 128, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(128, 128, feature_size, 3, 1, 1, batch_size)
        # forth layer (layer3)
        feature_size = feature_size // 2
        for i in range(4):
            cycle = cycle + conv2d(128, 256, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(256, 256, feature_size, 3, 1, 1, batch_size)
        # fifth layer (layer4)
        feature_size = feature_size // 2
        for i in range(4):
            cycle = cycle + conv2d(256, 512, feature_size, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(512, 512, feature_size, 3, 1, 1, batch_size)
        # seventh layer (fc)
        feature_size = feature_size // 2
        cycle = cycle + linear(512 * feature_size * feature_size, 4096, batch_size)
        cycle = cycle + linear(4096, 4096, batch_size)
        cycle = cycle + linear(4096, output_size, batch_size)
        cycle = cycle + nonlinear(output_size, batch_size)
        print_utilization(cycle, batch_size)
        print('The total cycle for the VGG19 is: ', cycle)
        print('The total time for the VGG19 is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')
    elif model == 'alexnet':
        cycle = 0
        # simulate for a alexnet
        # first layer (conv1)
        cycle = cycle + conv2d(3, 64, input_size, 11, 4, 2, batch_size)
        #print('first layer: ', cycle)
        # second layer (layer1)
        for i in range(1):
            cycle = cycle + conv2d(64, 192, 27, 5, 1, 2, batch_size)
            cycle = cycle + conv2d(192, 384, 13, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(384, 256, 13, 3, 1, 1, batch_size)
            cycle = cycle + conv2d(256, 256, 13, 3, 1, 1, batch_size)
        # seventh layer (fc)
        cycle = cycle + linear(256, output_size, batch_size)
        cycle = cycle + nonlinear(output_size, batch_size)
        print_utilization(cycle, batch_size)
        print('The total cycle for the alexnet is: ', cycle)
        print('The total time for the alexnet is: ', cycle / crossbar_frequency / crossbar_cycle / 1e6, 'ms')

if __name__ == "__main__":

    main()