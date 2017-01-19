# dump the `cat` content to a swap file
# pass that swap file for actual word segmentation
# 
# September, 2014
# Shuoyang Ding @ Johns Hopkins University

workdir=$2
swap_file_name=$2"/STDIN"`date +%s`".swap"

while read LINE; do
	echo ${LINE} >> $swap_file_name
done

script=$1
format=$3
shift
shift
shift
$script $format $swap_file_name $@
