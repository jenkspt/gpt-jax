Jax GPT
=======

This is a work-in-progress rewrite of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) in Jax/Flax.

One of the goals of this project is to try out [`jax.experimental.pjit`](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html). I'm curious about the performance differences for model size and distribution configurations.


## Prepare OpenWebText

Clone gpt-jax
```shell
git clone https://github.com/jenkspt/gpt-jax.git
cd gpt-jax
```

Install python dependencies
```shell
pip install -U pip
pip install tqdm
pip install numpy
pip install tiktoken
pip install datasets
pip install tensorflow
```

Prepare data
```shell
python data/openwebtext/prepare.py
```

This will generate the following files:  
`train_0.tfrecord`, `train_1.tfrecord` ... `train_{num_shards}`  
`val_0.tfrecord`, `val_1.tfrecord` ... `val_{num_shards}`


## Train (TPU single process)

Create TPU instance
```shell
ZONE=us-central-1f
TPU_TYPE=v2-8
VM_NAME=jax-gpt

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version v2-tf-stable \
```

SSH into TPU instance
```shell
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE 
```

Clone gpt-jax
```shell
git clone https://github.com/jenkspt/gpt-jax.git
cd gpt-jax
```

Install python dependencies
```shell
pip install -U pip
pip install tyro
pip install wandb
pip install -U tensorflow
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax
```

```shell
python3 train.py --help
```