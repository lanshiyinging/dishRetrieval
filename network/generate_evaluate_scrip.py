
for i in range(432):
    if i <= 146:
        comm = "get%d = `python get_hashcode_v2ep_double.py %d 8`" % (i, i)
    elif i >= 288:
        comm = "get%d = `python get_hashcode_v2ep_double.py %d 24`" % (i, i)
    else:
        comm = "get%d = `python get_hashcode_v2ep_double.py %d 12`" % (i, i)
    comm1 = "retrieval%d = `python retrieval_v2ep.py %d >> mAP.txt`" % (i, i)
    comm2 = "retrieval%d = `python retrieval_v2ep_hm.py %d >> mAP.txt`" % (i, i)
    with open("run_v4.sh", 'a') as f:
        f.write(comm + '\n' + comm1 + '\n' + comm2 + '\n')
