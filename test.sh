#!/bin/bash
#$ -q g.q
#$ -M dings@jhu.edu
#$ -l 'arch=*64,gpu=1,hostname=b1[12345678]*'
#$ -o /home/shuoyangd/experiments/nmt16-en-chn/exp/outs/ -e /home/shuoyangd/experiments/nmt16-en-chn/exp/outs/

print_usage () {
	printf "test.sh: test a nmt model with nematus

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

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/opt/NVIDIA/cuda-7/bin:/opt/NVIDIA/cuda-7.0/bin:$PATH

mkdir -p test

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$NEMATUS
mosesdecoder=$MOSES
stanford_seg=/home/shuoyangd/stanford-seg

# theano device
export n_gpus=`lspci | grep -i "nvidia" | wc -l`
export device=gpu`nvidia-smi | sed -e '1,/Processes/d' | tail -n+3 | head -n-1 | perl -ne 'next unless /^\|\s+(\d)\s+\d+/; $a{$1}++; for(my $i=0;$i<$ENV{"n_gpus"};$i++) { if (!defined($a{$i})) { print $i."\n"; last; }}' | tail -n 1`
echo $device

# apply tokenization
for prefix in $TST_PREFIX
do
  if [ $SRC == "zh" ] || [ $SRC == "chn"]  || [ $SRC == "cn" ]; then
    cat data/$prefix.$SRC_LANG | \
        ./segmentstd.sh $stanford_seg/segment.sh data/ ctb UTF-8 0 | \
        $mosesdecoder/scripts/tokenizer/escape-special-chars.perl | \
        ./chinese-punctuations-utf8.perl > data/$prefix.tok.$SRC_LANG
  else
   cat data/$prefix.$SRC_LANG | \
        $MOSES/scripts/tokenizer/normalize-punctuation.perl -l $SRC_LANG | \
        $MOSES/scripts/tokenizer/tokenizer.perl -a -l $SRC_LANG > data/$prefix.tok.$SRC_LANG
  fi
done

# apply truecase on test set
for prefix in $TST_PREFIX
do
  $MOSES/scripts/recaser/truecase.perl -model model/truecase-model.$SRC_LANG < data/$prefix.tok.$SRC_LANG > data/$prefix.tc.$SRC_LANG
done

# apply bpe
for prefix in $TST_PREFIX
do
  $SUBWORD/apply_bpe.py -c model/$SRC_LANG$TGT_LANG.bpe < data/$prefix.tc.$SRC_LANG > data/$prefix.bpe.$SRC_LANG
done

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m model/model.npz \
     -i $TSTDATA.bpe.$SRC_LANG \
     -o $TSTDATA.output \
     -k 12 -n -p 1

cat $TSTDATA.output | ./postprocess-test.sh > $TSTDATA.output.postprocessed
$MOSES/scripts/ems/support/wrap-xml.perl $TGT_FULL_LANG $WRAP_TEMPLATE < $TSTDATA.output.postprocessed > $TSTDATA.output.postprocessed.sgm

$MOSES/scripts/generic/mteval-v13a.pl -s $WRAP_TEMPLATE -r $TSTDATA.$TGT_LANG -t $TSTDATA.output.postprocessed.sgm > test/BLEU
$MOSES/scripts/generic/mteval-v13a.pl -c -s $WRAP_TEMPLATE -r $TSTDATA.$TGT_LANG -t $TSTDATA.output.postprocessed.sgm > test/BLEU-c

