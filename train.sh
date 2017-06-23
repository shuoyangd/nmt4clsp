#!/bin/bash

# default
CONFIG=""

print_usage () {
	printf "train.sh: train a nmt model with nematus

USAGE:
-c|--config    path to the env.sh config file
";
}

# Use > 1 to consume two arguments per pass in the loop (e.g. each
# argument has a corresponding value to go with it).
# Use > 0 to consume one or more arguments per pass in the loop (e.g.
# some arguments don't have a corresponding value to go with it such
# as in the --default example).
# note: if this is set to > 0 the /etc/hosts part is not recognized ( may be a bug )
while [[ $# > 0 ]]
do
key="$1"

case $key in
	-c|--config)
	CONFIG="$2"
	shift # past argument
	;;
	*)
	# unknown options
	print_usage
	exit
	;;
esac
shift # past argument or value
done

source $CONFIG
cd $WORKDIR

# theano device
export n_gpus=`lspci | grep -i "nvidia" | wc -l`
export device=`nvidia-smi | sed -e '1,/Processes/d' | tail -n+3 | head -n-1 | perl -ne 'next unless /^\|\s+(\d)\s+\d+/; $a{$1}++; for(my $i=0;$i<$ENV{"n_gpus"};$i++) { if (!defined($a{$i})) { print $i."\n"; last; }}' | tail -n 1`
echo gpu$device
if [ -z $device ] ; then
  echo "no device! grid cheaaaaaaaaaatin!"
  exit;
fi

cmd="python $ONMT/train.py -data data/$TRN_PREFIX+$DEV_PREFIX.bin.train.pt -save_model model/model -gpus $device"

if [ ! -z $TRAIN_TRAIN_FROM ] ; then
  cmd=$cmd" -train_from $TRAIN_FROM"
fi

if [ ! -z $TRAIN_TRAIN_FROM_STATE_DICT ] ; then
  cmd=$cmd" -train_from $TRAIN_FROM_STATE_DICT"
fi

if [ ! -z $TRAIN_START_EPOCH ] ; then
  cmd=$cmd" -start_epoch $START_EPOCH"
fi

if [ ! -z $TRAIN_PRE_WORD_VECS_ENC ] ; then
  cmd=$cmd" -pre_word_vecs_enc $TRAIN_PRE_WORD_VECS_ENC"
fi

if [ ! -z $TRAIN_PRE_WORD_VECS_DEC ] ; then
  cmd=$cmd" -pre_word_vecs_dec $TRAIN_PRE_WORD_VECS_DEC"
fi

if [ ! -z $TRAIN_BRNN ] ; then
  cmd=$cmd" -brnn" 
fi

cmd=$cmd" -layers $TRAIN_LAYERS -rnn_size $TRAIN_RNN_SIZE -word_vec_size $TRAIN_WORD_VEC_SIZE -batch_size $TRAIN_BATCH_SIZE -epochs $TRAIN_EPOCHS -optim $TRAIN_OPTIM -dropout $TRAIN_DROPOUT -learning_rate $TRAIN_LEARNING_RATE"
echo $cmd
$cmd

