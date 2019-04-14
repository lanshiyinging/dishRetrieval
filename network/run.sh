#!/bin/sh

python dsh_dishNet.py 1>net.log 2>net.err
python dsh_dishNet_change_pre.py 1>net1.log 2>net1.err
python dsh_dishNet_op_adam.py 1>net2.log 2>net2.err
python dsh_dishNet_op_gdo.py 1>net3.log 2>net3.err
python dsh_dishNet_op_mom.py 1>net4.log 2>net4.err
python dsh_dishNet_lr1.py 1>net5.log 2>net5.err
python dsh_dishNet_lr2.py 1>net6.log 2>net6.err
python dsh_dishNet_lr3.py 1>net7.log 2>net7.err
python dsh_dishNet_dp1.py 1>net8.log 2>net8.err
python dsh_dishNet_dp2.py 1>net9.log 2>net9.err
python dsh_dishNet_dp3.py 1>net10.log 2>net10.err
python get_hashcode_v2ep.py 0 > get0.log
python get_hashcode_v2ep.py 1 > get1.log
python get_hashcode_v2ep.py 2 > get2.log
python get_hashcode_v2ep.py 3 > get3.log
python get_hashcode_v2ep.py 4 > get4.log
python get_hashcode_v2ep.py 5 > get5.log
python get_hashcode_v2ep.py 6 > get6.log
python get_hashcode_v2ep.py 7 > get7.log
python get_hashcode_v2ep.py 8 > get8.log
python get_hashcode_v2ep.py 9 > get9.log
python get_hashcode_v2ep.py 10 > get10.log
python retrieval_v2ep.py 0 >> mAP.txt
python retrieval_v2ep.py 1 >> mAP.txt
python retrieval_v2ep.py 2 >> mAP.txt
python retrieval_v2ep.py 3 >> mAP.txt
python retrieval_v2ep.py 4 >> mAP.txt
python retrieval_v2ep.py 5 >> mAP.txt
python retrieval_v2ep.py 6 >> mAP.txt
python retrieval_v2ep.py 7 >> mAP.txt
python retrieval_v2ep.py 8 >> mAP.txt
python retrieval_v2ep.py 9 >> mAP.txt
python retrieval_v2ep.py 10 >> mAP.txt


