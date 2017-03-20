import numpy
import os
import sys

sys.path.append(os.environ['NEMATUS'] + "/nematus")

VOCAB_SIZE = int(os.environ['VOCAB_SIZE'])
SRC = os.environ['SRC_LANG']
TGT = os.environ['TGT_LANG']
DATA_DIR = "data"

from nmt import train

if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    dim_word=500,
                    dim=1024,
                    n_words=VOCAB_SIZE,
                    n_words_src=VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=50,
                    batch_size=80,
                    valid_batch_size=80,
                    datasets=[DATA_DIR + "/" + os.environ['TRN_PREFIX'] +'.bpe.' + SRC, DATA_DIR + "/" + os.environ['TRN_PREFIX'] + '.bpe.' + TGT],
                    valid_datasets=[DATA_DIR + "/" + os.environ['DEV_PREFIX'] + '.bpe.' + SRC, DATA_DIR + "/" + os.environ['DEV_PREFIX'] + '.bpe.' + TGT],
                    dictionaries=[DATA_DIR + "/" + os.environ['TRN_PREFIX'] + '.bpe.' + SRC + '.json',DATA_DIR + "/" + os.environ['TRN_PREFIX'] + '.bpe.' + TGT + '.json'],
                    validFreq=5000,
                    dispFreq=1000,
                    saveFreq=10000,
                    sampleFreq=5000,
                    use_dropout=False,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script=os.environ['QSUB_PREFIX'] + " " + os.environ['VLD_SCRPT'])
    print validerr
