#!/bin/sh

python dsh_dishNet_change_pre.py 1>net.log 2>net.err
python dsh_dishNet_change_pre1.py 1>net1.log 2>net1.err
python dsh_dishNet_change_pre2.py 1>net2.log 2>net2.err
#python dsh_dishNet_change_pre3.py 1>net3.log 2>net3.err
