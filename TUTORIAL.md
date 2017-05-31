# Tutorial

This tutorial was written for [JHU NMT Winter School 2017](http://statmt.org/jhu/?n=NMTWinterSchool.HomePage). It serves as a step-by-step guide to setup neural MT bechmark on CLSP grid (which runs Debian Linux). It does not attempt to be system-agnostic so beware if you are trying to do this on your own machine.

This tutorial uses [nematus](https://github.com/shuoyangd/nematus) and [amunmt](https://github.com/emjotde/amunmt) as the neural MT architecture. Nematus is a [theano](https://github.com/Theano/Theano)-based software that builds neural MT models and also decodes with them. AMU-NMT is a neural MT decoder that takes a model trianed by nematus. It is written in C++ so the decoding speed is said to be faster.

The data and configurations in this repository allows you to build a Chinese transliteration model (Chinese Charater to Pinyin) with the NMT architecture. This task is much easier than translation so your model could yield reasonable output with much smaller data and shorter training time.

INSTRUCTIONS
------------

### Setting up Virtualenv

For your best, this repository assumes usage of [virtualenv](https://virtualenv.pypa.io/en/stable/). The basic idea for virtualenv is to create a standalone python environment so (1) you don't run the risk of mess up your root python environment and (2) you no longer need root privilege on the grid. If you do not use it, start by getting it with:

```
pip install virtualenv
```

And then, set up a virtual environment wherever you want, and activate it:

```
virtualenv /path/to/wherever
source /path/to/wherever/bin/activate
```

Sometimes this gives you an out-dated version of `pip`. Make sure you get the latest version:

```
pip install --upgrade pip
```

To leave this virtual environment, simply type `deactivate`. But don't do it now, as we are going to install a series of packages in this environment. Whenever you need to access these packages, you need to enter this environment again.

### Setting up Theano

As of 05/31/2017, the best practice seems to use the latest theano release, but as later release come out, this may no longer apply. To get the latest theano release:

```
pip install theano
```

### Setting up Nematus

Please use this copy: [https://github.com/shuoyangd/nematus](https://github.com/shuoyangd/nematus) -- it contains a tiny tweak with `Popen` that enables the validation script to be submitted normally. DO NOT use upstream on CLSP grid at least for now.

First, let's make sure we have the latest `ipython`, as we would be using `ipdb` and some older version of `ipython` has some problems with that:

```
pip install ipython --upgrade
```

Then install a bunch of dependencies:

```
pip install numpy numexpr cython tables ipdb
python setup.py install
```

You may experience the following problem when installing tables:

> ERROR:: Could not find a local HDF5 installation

The most straight-forward solution is to install HDF5 by yourself. Alternatively you can also use my installation by adding the following lines to `.bash_profile`:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/shuoyangd/local/hdf5/lib"
export HDF5_DIR="/home/shuoyangd/local/hdf5"
```

### Checking out Scripts

Checkout [subword-nmt](https://github.com/rsennrich/subword-nmt) and [mosesdecoder](https://github.com/moses-smt/mosesdecoder). You don't need to compile any of them because we are just using scripts.

Checkout this repository as well. `cd` into the directory you checked out this repo, you will spend most of your time in this directory from now on.

### Configuring Nematus

`.nematusrc` supposedly contains all of the files and software you need to run Nematus. Follow the comments in that file and change them to your directories.

`config.py` contains some hyperparameters (e.g. `batch_size` `maxlen`) which you may want to check before running experiments. Specially for the transliteration task, because all the training data are lexicon, you may want to increase `maxlen` for real translation task.

### Preprocessing

Run `./preprocess.sh --config /path/to/.nematusrc` and it will take care of everything for you. You don't need GPU for this.

### Training

> **!Caution!**: make sure you always either `qsub` or `qlogin` and turn on `-l gpu=1` when you use GPU!

Run the following command to submit your training job to the gpu queue.

```
qsub -q g.q -l gpu=1 -m ae -M email@jhu.edu -o /path/to/stdout -e /path/to/stderr train.sh --config /path/to/.nematusrc
```

Model will be dumped into `$WORKDIR/model` and validation will be run automatically as separate jobs as training proceeds. `$WORKDIR` is defined as in `.nematusrc`

> I sometimes see something like the following in the error output:
> 
> ```
> Traceback (most recent call last):
>   File "config.py", line 12, in <module>
>     from nmt import train
>   File "/home/shuoyangd/nmt/nematus/nematus/nmt.py", line 6, in <module>
>     import theano
>   File "/home/shuoyangd/pyenv/theano/local/lib/python2.7/site-packages/theano/__init__.py", line 110, in <module>
>     import theano.sandbox.cuda
>   File "/home/shuoyangd/pyenv/theano/local/lib/python2.7/site-packages/theano/sandbox/cuda/__init__.py", line 708, in <module>
>     use(device=config.device, force=config.force_device, test_driver=False)
>   File "/home/shuoyangd/pyenv/theano/local/lib/python2.7/site-packages/theano/sandbox/cuda/__init__.py", line 554, in use
>     assert active_device == device, (active_device, device)
> AssertionError: (3, 1)
> ```
>
> But generally re-submitting and/or cleaning the theano cache resolves it. Hence I never dive too deep into the problem. Someone may know this better than I do.

The training will take a while. At the same time, let's set up AMU-NMT for fast neural MT decoding.

### Setting up AMU-NMT

I will use exact tested dependency version as specified on AMU-NMT github page. Later version might work as well, but not for sure.

#### boost

First of all, you need boost library verion 1.63. It takes a while to set up your own. If you want to use mine, add the following lines to your `.bash_profile`: 

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/shuoyangd/local/lib"
export LIBRARY_PATH=$LIBRARY_PATH:"/home/shuoyangd/local/lib"
export BOOST_ROOT="/home/shuoyangd/local/pkg/boost_1_63_0"
```

The default boost on CLSP seems to be version 1.55 and does not work for me.

#### cmake

You need cmake version 3.5.1. Get it [here](https://cmake.org/files/). 

```
tar -zxvf cmake-3.5.1.tar.gz
cd cmake-3.5.1
./configure --prefix=/path/to/install/location
make
make install
```

Adding prefix directory to boostrap script will enable you to compile on the grid without root privileges.

#### Compile AMU-NMT

Because `cuda` library only exists on those machines that have a GPU, you first need to qlogin to a machine with gpu, e.g. `qlogin -l hostname=b*`.

Make sure you already configured cuda-related environment variables properly. Mine looks like this:

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
```

You can setup CuMeM and/or CuDNN for better performance and smaller memory usage. But it's optional.

Checkout [this](https://github.com/emjotde/amunmt) repo. After that, compile as documented:

```
mkdir build
cd build
cmake ..
make -j
```

Your final build should reside in `/path/to/clone/build/bin/amun`.

### Decode with AMU-NMT

Check out the working directory you set up in `.nematusrc` for a `$WORKDIR/model/model.npz` file. This is where Nematus will write the latest model during training. If you don't see it yet, you need to wait longer for the training to proceed.

When you ready to run some decoding, edit `config.yml` in this repository to pass necessary information to AMU-NMT. Everything should be self-evident.

> There is only one important thing to note: if you don't plan to use GPU, always set `gpu-threads` to 0. If you are using it, make sure you always either `qsub` or `qlogin` and turn on `-l gpu=1` when you use GPU.

To run AMU-NMT on GPU, use the following command:

```
echo "cd /path/to/repo && cat data/lexicon.tst.zhch | ~/nmt/amunmt/build/bin/amun -c config.yml" | qsub -q g.q -l gpu=1 -m ae -M email@jhu.edu -o /path/to/stdout -e /path/to/stderr
```

> I should run tokenize, bpe and everything before doing this. This does not matter much for this task, but sometime soon I need to integrate AMU-NMT into `test.sh`.

### Decode with Nematus

As long as you have set up `.nematusrc`, you can use `test.sh` to decode with Nematus as well. Just run `./test.sh --config /path/to/.nematusrc`


