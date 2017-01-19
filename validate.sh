#!/bin/sh

# default
CONFIG=""

print_usage () {
	printf "validate.sh: validate a nmt model with nematus (for early-stopping)

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

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$NEMATUS

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$MOSES

# theano device
nvidia-smi
export n_gpus=`lspci | grep -i "nvidia" | wc -l`
export device=gpu`nvidia-smi | sed -e '1,/Processes/d' | tail -n+3 | head -n-1 | perl -ne 'next unless /^\|\s+(\d)\s+\d+/; $a{$1}++; for(my $i=0;$i<$ENV{"n_gpus"};$i++) { if (!defined($a{$i})) { print $i."\n"; last; }}' | tail -n 1`
# export device=gpu`/home/gkumar/scripts/free-gpu`
#`nvidia-smi | grep -B 1 ' 0%' | head -1 | cut -d\  -f4`
echo "validate on $device of host "`hostname`
#model prefix
prefix=$WORKDIR/model/model.npz

dev=$DEVDATA.bpe.$SRC_LANG
ref=$DEVDATA.tok.$TGT_LANG

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev \
     -o $dev.output.dev \
     -k 12 -n -p 1

./postprocess-dev.sh < $dev.output.dev > $dev.output.postprocessed.dev


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed.dev >> ${prefix}_bleu_scores
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
fi
