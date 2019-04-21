#!/bin/sh

out1 = `python dsh_dishNet_op_adam1.py 1>net11.log 2>net11.err`
out2 = `python get_hashcode_v2ep.py 11 > get11.log`
out3 = `python get_hashcode_v2ep_all.py 8 > get2_all.log`
out4 = `python get_hashcode_v2ep_all.py 11 > get11_all.log`
