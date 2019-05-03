#!/usr/bin/sh

python train.py > train.log
var = $?
while [ $var -nq 0 ]; do
    python train.py > train.log
    var = $?
done