import torch
from transformers import BertModel, BertTokenizer
import time
import string
import random
from time import sleep

def generate_random_string(length=8):
    # Define the character set: letters (upper and lower), digits, and punctuation.
    characters = string.ascii_letters + string.digits + string.punctuation
    # Generate a random string by selecting a random character for each position.
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def benchmark_bert_inference(
    device: str = 'cpu',
    batch_size: int = 32,
    seq_length: int = 128,
    iterations: int = 100,
    warmup: int = 10
):
    # Load pre-trained BERT model and tokenizer.
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()  # set the model to evaluation mode

    # Create dummy text and tokenize it.
    dummy_text = "Hello, how are you doing today?"
    dummy_texts = [dummy_text] * batch_size
    for i in range(batch_size):
        # generate a random string of words
        dummy_texts[i] = generate_random_string()
        dummy_texts[i] = " ".join([dummy_texts[i]] * (seq_length // 6))

    # Repeat the text to roughly match the desired sequence length.

    inputs = tokenizer(
        dummy_texts,
        return_tensors='pt',
        max_length=seq_length,
        padding='max_length',
        truncation=True
    )
    # Move inputs to the specified device.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warm-up phase: run a few inferences to stabilize latency.
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)

    # Benchmark inference latency.
    times = []
    for _ in range(iterations):
        if device == 'cuda':
            torch.cuda.synchronize()  # ensure all previous CUDA ops are finished
        start_time = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        if device == 'cuda':
            torch.cuda.synchronize()  # wait for the inference to finish on GPU
        elapsed = time.time() - start_time
        times.append(elapsed)
        sleep(0.1)  # small sleep to avoid overwhelming the GPU

    avg_latency = sum(times) / len(times)
    print(f"Average inference latency over {iterations} iterations: {avg_latency * 1000:.2f} ms on {device}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmark_bert_inference(device=device)