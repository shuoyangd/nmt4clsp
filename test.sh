#!/bin/bash

print_usage () {
	printf "test.sh: test a nmt model with nematus

USAGE:
-c|--config    path to the env.sh config file
-d|--decoder   choose decoder for testing
-s|--scorer    choose a scorer for evaluation [nist|multi-bleu]
--multi-ref    test set has multiple reference on the target side (default = false)
";
}

MULTIREF=false
SCORER="multi-bleu"

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
        -d|--decoder)
        DECODER="$2"
        shift
        ;;
        -s|--scorer)
        SCORER="$2"
        shift
        ;;
        --multi-ref)
        MULTIREF=true
        shift
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

mkdir -p test

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$NEMATUS
mosesdecoder=$MOSES
stanford_seg=/home/shuoyangd/stanford-seg

# theano device
export n_gpus=`lspci | grep -i "nvidia" | wc -l`
export device=`nvidia-smi | sed -e '1,/Processes/d' | tail -n+3 | head -n-1 | perl -ne 'next unless /^\|\s+(\d)\s+\d+/; $a{$1}++; for(my $i=0;$i<$ENV{"n_gpus"};$i++) { if (!defined($a{$i})) { print $i."\n"; last; }}' | tail -n 1`
echo $device

# apply tokenization
for prefix in $TST_PREFIX
do
  if [ "$SRC_LANG" == "zh" ] || [ "$SRC_LANG" == "chn" ]  || [ "$SRC_LANG" == "cn" ] ; then
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

model=$WORKDIR/model/model.ensemble.npz
cp $WORKDIR/model/model.npz.json ${model}.json
tst=$TSTDATA.bpe.$SRC_LANG
ref=$TSTDATA.$TGT_LANG

if [ $SCORER == "multi-bleu" ] ; then
  $MOSES/scripts/ems/support/reference-from-sgm.perl $TSTDATA.$TGT_LANG $WRAP_TEMPLATE $TSTDATA.${TGT_LANG}.txt
  if [ $MULTIREF == false ]; then
    if [ "$TGT_LANG" == "zh" ] || [ "$TGT_LANG" == "chn" ] || [ "$TGT_LANG" == "cn" ] ; then
      cat ${TSTDATA}.${TGT_LANG}.txt | \
          ./segmentstd.sh $stanford_seg/segment.sh data/ ctb UTF-8 0 | \
          $mosesdecoder/scripts/tokenizer/escape-special-chars.perl | \
          ./chinese-punctuations-utf8.perl > data/${TST_PREFIX}.${TGT_LANG}.tok
    else
      cat ${TSTDATA}.${TGT_LANG}.txt | \
          $MOSES/scripts/tokenizer/normalize-punctuation.perl -l $TGT_LANG | \
          $MOSES/scripts/tokenizer/tokenizer.perl -a -l $TGT_LANG > data/${TST_PREFIX}.${TGT_LANG}.tok
    fi
  else
    if [ "$TGT_LANG" == "zh" ] || [ "$TGT_LANG" == "chn" ] || [ "$TGT_LANG" == "cn" ] ; then
      $MOSES/scripts/ems/support/run-command-on-multiple-refsets.perl 'cat mref-input-file | \
          ./segmentstd.sh $stanford_seg/segment.sh data/ ctb UTF-8 0 | \
          $mosesdecoder/scripts/tokenizer/escape-special-chars.perl | \
          ./chinese-punctuations-utf8.perl > mref-output-file' data/${TST_PREFIX}.${TGT_LANG}.txt \
          data/${TST_PREFIX}.${TGT_LANG}.tok
    else
      $MOSES/scripts/ems/support/run-command-on-multiple-refsets.perl 'cat mref-input-file | \
          $MOSES/scripts/tokenizer/normalize-punctuation.perl -l $TGT_LANG | \
          $MOSES/scripts/tokenizer/tokenizer.perl -a -l $TGT_LANG' data/${TST_PREFIX}.${TGT_LANG}.txt \
          data/${TST_PREFIX}.${TGT_LANG}.tok
    fi
  fi
fi

# decode
if [ $DECODER == "nematus" ] ; then
  THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $model \
     -i $tst \
     -o $tst.output \
     -k 12 -n -p 1
elif [ $DECODER == "amunmt" ] ; then
  rand=`od -vAn -N4 -tu4 < /dev/urandom | sed 's/ //g'`
  amu_config=config.$rand.yml

  touch $amu_config 
  echo """relative-paths: $RELATIVE_PATHS

beam-size: $BEAM_SIZE
devices: [$device]
normalize: $NORMALIZE
gpu-threads: 0
cpu-threads: 1

scorers: 
  F0: 
    path: $model
    type: Nematus
    
weights:
  F0: 1.0

source-vocab: data/$TRN_PREFIX.bpe.$SRC_LANG.json
target-vocab: data/$TRN_PREFIX.bpe.$TGT_LANG.json""" > $amu_config

  if [ ! -z $BPE ] ; then
    echo "bpe: $BPE" >> $amu_config
  fi
  echo "debpe: $DEBPE" >> $amu_config

  $AMUNMT/build/bin/amun -c $amu_config < $tst > $tst.output
  rm $amu_config
else
  echo "decoder not suppported" >&2
  exit 1
fi

cat $tst.output | ./postprocess-test.sh > $tst.output.postprocessed

if [ $SCORER == "nist" ] ; then
  $MOSES/scripts/ems/support/wrap-xml.perl $TGT_FULL_LANG $WRAP_TEMPLATE < $tst.output.postprocessed > $tst.output.postprocessed.sgm
  $MOSES/scripts/generic/mteval-v13a.pl -s $WRAP_TEMPLATE -r $TSTDATA.$TGT_LANG -t $tst.output.postprocessed.sgm > test/BLEU
  $MOSES/scripts/generic/mteval-v13a.pl -c -s $WRAP_TEMPLATE -r $TSTDATA.$TGT_LANG -t $tst.output.postprocessed.sgm > test/BLEU-c
elif [ $SCORER == "multi-bleu" ] ; then
  $MOSES/scripts/generic/multi-bleu.perl -lc data/${TST_PREFIX}.${TGT_LANG}.tok < $tst.output.postprocessed > test/BLEU
  $MOSES/scripts/generic/multi-bleu.perl data/${TST_PREFIX}.${TGT_LANG}.tok < $tst.output.postprocessed > test/BLEU-c
fi

