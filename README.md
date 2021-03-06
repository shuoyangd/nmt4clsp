# nmt4clsp

**Nov. 8, 2017**: Now that I'm no longer using Nematus and writing plain bash script to run experiments, so consider this project deprecated. If you are interested in finding out what I use now, check this new project called [tape4nmt](https://github.com/shuoyangd/tape4nmt).

This directory is forked from Rico Sennrich's [wmt16-scripts](https://github.com/rsennrich/wmt16-scripts), with tweaks to facilitate easier usage and special configurations added to train a Chinese-English neural machine translation system. Note that apart from the preprocess script, all the other scripts are language-agnostic.

All the scripts are tested to run on the CLSP grid of Johns Hopkins University.

DEPENDENCIES
------------
+ [theano](https://github.com/Theano/Theano)
+ [nematus](https://github.com/shuoyangd/nematus) (Note the upstream copy may not run smoothly on CLSP grid)
+ [subword-nmt](https://github.com/rsennrich/subword-nmt)
+ [Moses Decoder](https://github.com/shuoyangd/mosesdecoder)

### (Optional) Chinese
+ Stanford Chinese Word Segmentor >= v3.4.1

### (Optional) AMUNMT
+ [amunmt](https://github.com/emjotde/amunmt)

INSTRUCTIONS
------------

See [Tutorial](https://github.com/shuoyangd/nmt4clsp/blob/master/TUTORIAL.md).

