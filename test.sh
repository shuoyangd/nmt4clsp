#!/bin/bash

print_usage () {
	printf "test.sh: test a nmt model with opennmt

USAGE:
-c|--config    path to the env.sh config file
-s|--scorer    choose a scorer for evaluation [nist|multi-bleu]
--multi-ref    test set has multiple reference on the target side (default = false)
--model        the model to be tested
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
        -s|--scorer)
        SCORER="$2"
        shift
        ;;
        --multi-ref)
        MULTIREF=true
        shift
        ;;
        --model)
        MODEL="$2"
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

model=$MODEL
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
          $MOSES/scripts/tokenizer/tokenizer.perl -a -l $TGT_LANG > mref-output-file' data/${TST_PREFIX}.${TGT_LANG}.txt \
          data/${TST_PREFIX}.${TGT_LANG}.tok
    fi
  fi
fi

# decode
python $ONMT/translate.py -gpu $device -model $MODEL -src $tst -tgt $ref -replace_unk -verbose -output $tst.output

cat $tst.output | ./postprocess-test.sh > $tst.output.postprocessed

if [ $SCORER == "nist" ] ; then
  $MOSES/scripts/ems/support/wrap-xml.perl $TGT_FULL_LANG $WRAP_TEMPLATE < $tst.output.postprocessed > $tst.output.postprocessed.sgm
  $MOSES/scripts/generic/mteval-v13a.pl -s $WRAP_TEMPLATE -r $TSTDATA.$TGT_LANG -t $tst.output.postprocessed.sgm > test/BLEU
  $MOSES/scripts/generic/mteval-v13a.pl -c -s $WRAP_TEMPLATE -r $TSTDATA.$TGT_LANG -t $tst.output.postprocessed.sgm > test/BLEU-c
elif [ $SCORER == "multi-bleu" ] ; then
  $MOSES/scripts/generic/multi-bleu.perl -lc data/${TST_PREFIX}.${TGT_LANG}.tok < $tst.output.postprocessed > test/BLEU
  $MOSES/scripts/generic/multi-bleu.perl data/${TST_PREFIX}.${TGT_LANG}.tok < $tst.output.postprocessed > test/BLEU-c
fi

