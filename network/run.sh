#!/bin/sh

out1 = `python dsh_dishNet.py 1>net.log 2>net.err`
out2 = `python dsh_dishNet_change_pre.py 1>net1.log 2>net1.err`
out3 = `python dsh_dishNet_op_adam.py 1>net2.log 2>net2.err`
out4 = `python dsh_dishNet_op_gdo.py 1>net3.log 2>net3.err`
out5 = `python dsh_dishNet_op_mom.py 1>net4.log 2>net4.err`
out6 = `python dsh_dishNet_lr1.py 1>net5.log 2>net5.err`
out7 = `python dsh_dishNet_lr2.py 1>net6.log 2>net6.err`
out8 = `python dsh_dishNet_lr3.py 1>net7.log 2>net7.err`
out9 = `python dsh_dishNet_dp1.py 1>net8.log 2>net8.err`
out10 = `python dsh_dishNet_dp2.py 1>net9.log 2>net9.err`
out11 = `python dsh_dishNet_dp3.py 1>net10.log 2>net10.err`
out12 = `python get_hashcode_v2ep.py 0 > get0.log`
out13 = `python get_hashcode_v2ep.py 1 > get1.log`
out14 = `python get_hashcode_v2ep.py 2 > get2.log`
out15 = `python get_hashcode_v2ep.py 3 > get3.log`
out16 = `python get_hashcode_v2ep.py 4 > get4.log`
out17 = `python get_hashcode_v2ep.py 5 > get5.log`
out18 = `python get_hashcode_v2ep.py 6 > get6.log`
out19 = `python get_hashcode_v2ep.py 7 > get7.log`
out20 = `python get_hashcode_v2ep.py 8 > get8.log`
out21 = `python get_hashcode_v2ep.py 9 > get9.log`
out22 = `python get_hashcode_v2ep.py 10 > get10.log`
out23 = `python retrieval_v2ep.py 0 > mAP.txt`
out24 = `python retrieval_v2ep.py 1 >> mAP.txt`
out25 = `python retrieval_v2ep.py 2 >> mAP.txt`
out26 = `python retrieval_v2ep.py 3 >> mAP.txt`
out27 = `python retrieval_v2ep.py 4 >> mAP.txt`
out28 = `python retrieval_v2ep.py 5 >> mAP.txt`
out29 = `python retrieval_v2ep.py 6 >> mAP.txt`
out30 = `python retrieval_v2ep.py 7 >> mAP.txt`
out31 = `python retrieval_v2ep.py 8 >> mAP.txt`
out32 = `python retrieval_v2ep.py 9 >> mAP.txt`
out33 = `python retrieval_v2ep.py 10 >> mAP.txt`


