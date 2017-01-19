#/bin/sh

# suffix of target language files
lng=en

sed -r 's/\@\@ //g' | \
# sed -r 's/ \@(\S*?)\@ /\1/g' | \
./detruecase.perl | \
./detokenizer.perl -l $lng
