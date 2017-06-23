#!/bin/sh

# default
CONFIG=""
BPE_ONLY=""

print_usage () {
	printf "preprocess.sh: preprocess data for opennmt training

USAGE:
-c|--config    path to the .onmtrc config file
--bpe-only     only run bpe (no arg)
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
        --bpe-only)
        BPE_ONLY="yes"
        # no argument
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
mkdir -p data model

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation. 
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# and the file names (currently, data/corpus and data/newsdev2016 are being processed)

# in the tokenization step, you will want to remove Romanian-specific normalization / diacritic removal,
# and you may want to add your own.
# also, you may want to learn BPE segmentations separately for each language,
# especially if they differ in their alphabet

# suffix of source language files
SRC=$SRC_LANG

# suffix of target language files
TRG=$TGT_LANG

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=$BPE_OPT

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$MOSES

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$SUBWORD

# path to nematus ( https://www.github.com/rsennrich/nematus )
onmt=$ONMT

# path to stanford-seg
stanford_seg=/home/shuoyangd/stanford-seg


# tokenize
if [ -z $BPE_ONLY ] ; then 
  for prefix in $TRN_PREFIX $DEV_PREFIX
  do
    # segmentation for chinese
    if [ $SRC == "zh" ] || [ $SRC == "chn" ] || [ $SRC == "cn" ] ; then
      cat data/$prefix.$SRC | \
          ./segmentstd.sh $stanford_seg/segment.sh data/ ctb UTF-8 0 | \
          $mosesdecoder/scripts/tokenizer/escape-special-chars.perl | \
          ./chinese-punctuations-utf8.perl > data/$prefix.tok.$SRC
    else
      cat data/$prefix.$SRC | \
          $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
          $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > data/$prefix.tok.$SRC
    fi
  
  
    if [ $TRG == "zh" ] || [ $TRG == "chn" ]  || [ $TRG == "cn" ] ; then
      cat data/$prefix.$TRG | \
          ./segmentstd.sh $stanford_seg/segment.sh data/ ctb UTF-8 0 | \
          $mosesdecoder/scripts/tokenizer/escape-special-chars.perl | \
          ./chinese-punctuations-utf8.perl > data/$prefix.tok.$TRG
    else 
      cat data/$prefix.$TRG | \
          $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG | \
          $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > data/$prefix.tok.$TRG
    fi
  done
  
  # clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
  $mosesdecoder/scripts/training/clean-corpus-n.perl $TRNDATA.tok $SRC $TRG $TRNDATA.tok.clean 1 80
  
  # train truecaser
  $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $TRNDATA.tok.clean.$SRC -model model/truecase-model.$SRC
  $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $TRNDATA.tok.clean.$TRG -model model/truecase-model.$TRG
  
  # apply truecaser (cleaned training corpus)
  for prefix in $TRN_PREFIX
   do
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.clean.$SRC > data/$prefix.tc.$SRC
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$TRG < data/$prefix.tok.clean.$TRG > data/$prefix.tc.$TRG
   done

  # apply truecaser (dev/test files)
  for prefix in $DEV_PREFIX
   do
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.$SRC > data/$prefix.tc.$SRC
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$TRG < data/$prefix.tok.$TRG > data/$prefix.tc.$TRG
   done
else
  for prefix in $TRN_PREFIX $DEV_PREFIX
  do
    if [ ! -f data/$prefix.tc.$SRC ]; then
      cp data/$prefix.$SRC data/$prefix.tc.$SRC
    fi

    if [ ! -f data/$prefix.tc.$TRG ]; then
      cp data/$prefix.$TRG data/$prefix.tc.$TRG
    fi
  done
fi

# train BPE
cat data/$TRN_PREFIX.tc.$SRC data/$TRN_PREFIX.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe

# apply BPE
for prefix in $TRN_PREFIX $DEV_PREFIX
 do
  $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$SRC
  $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$TRG > data/$prefix.bpe.$TRG
  sed -i "$ d" data/$prefix.bpe.$SRC;
  sed -i "$ d" data/$prefix.bpe.$TRG;
done

# build binary training format
python $ONMT/preprocess.py -train_src data/$TRN_PREFIX.bpe.$SRC -train_tgt data/$TRN_PREFIX.bpe.$TRG -valid_src data/$DEV_PREFIX.bpe.$SRC -valid_tgt data/$DEV_PREFIX.bpe.$TRG -save_data data/$TRN_PREFIX+$DEV_PREFIX.bin

