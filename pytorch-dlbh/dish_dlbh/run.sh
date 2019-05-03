#!/usr/bin/sh

python train.py > train.log
var=$?
while [ $var -eq 1 ]; do
    python train.py > train.log
    var=$?
done