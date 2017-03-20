workdir=$2
rand=`od -vAn -N4 -tu4 < /dev/urandom | sed 's/ //g'`
swap_file_name=$2"/STDIN${rand}.swap"

cat | sed "s/[[:cntrl:]]//g" | sed "s/$/。/" >> "${swap_file_name}"

script=$1
format=$3
shift
shift
shift
$script $format $swap_file_name $@ | sed -e "s/。$//"
