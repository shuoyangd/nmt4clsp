#/bin/sh

sed -r 's/\@\@ //g' | \
./detruecase.perl
