#!/bin/bash

touch times.txt
sum_time=0

for a in 50000 #5000 15000 20000 25000
do
start=`date +%s`
python3 render.py --engine ../simulation --particles $a --duration 10 --speed 800 --output_folder "output$a"
end=`date +%s`

runtime=$((end-start))
echo "$a $runtime" >> times.txt
let sum_time=sum_time+runtime
done

echo "-----------------------------------------"
echo "\n ALL TIME : $sum_time seconds \n" >> times.txt
