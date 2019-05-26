
for i in range(288):
    if i <= 95:
        comm = "get%d = `python get_hashcode_v2ep_double.py %d 8`" % (i, i)
    elif i >= 192:
        comm = "get%d = `python get_hashcode_v2ep_double.py %d 24`" % (i, i)
    else:
        comm = "get%d = `python get_hashcode_v2ep_double.py %d 12`" % (i, i)
    comm1 = "retrieval%d = `python retrieval_v2ep.py %d >> mAP.txt`" % (i, i)
    comm2 = "retrieval%d = `python retrieval_v2ep_hm.py %d >> mAP.txt`" % (i, i)
    with open("run_v4.sh", 'a') as f:
        f.write(comm + '\n' + comm1 + '\n' + comm2 + '\n')


'''
model_list = [98, 201, 213, 288, 322, 332, 364, 428, 118, 242, 388]
for i in model_list:
    if i <= 146:
        comm = "get%d = `python get_hashcode_v2ep_double_mini.py %d 8`" % (i, i)
    elif i >= 288:
        comm = "get%d = `python get_hashcode_v2ep_double_mini.py %d 24`" % (i, i)
    else:
        comm = "get%d = `python get_hashcode_v2ep_double_mini.py %d 12`" % (i, i)
    comm1 = "retrieval%d = `python retrieval_v2ep_mini.py %d >> mAP_mini.txt`" % (i, i)
    comm2 = "retrieval%d = `python retrieval_v2ep_hm_mini.py %d >> mAP_mini.txt`" % (i, i)
    with open("run_v5.sh", 'a') as f:
        f.write(comm + '\n' + comm1 + '\n' + comm2 + '\n')
'''

