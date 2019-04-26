
k_list = [8, 12, 24]
lr_method_list = ["decay", "smooth"]
base_lr_list = [0.001, 0.0001, 0.00005, 0.000025, 0.00001]
dropout_list = [0.8, 0.7, 0.6, 0.5]
optimize_list = ["adam", "mome"]
alpha_list = [0, 0.001, 0.01, 0.1]
m_list = [6, 3, 2, 1]

num = 0
for k in k_list:
    for lr_method in lr_method_list:
        for base_lr in base_lr_list:
            for dropout in dropout_list:
                for optimize in optimize_list:
                    for alpha in alpha_list:
                        for m in m_list:
                            comm = "out%d = python dsh_dishNet_param.py %d %s %f %f %s %f %d %d" % (num, k, lr_method, base_lr, dropout, optimize, alpha, m, num)
                            comm = comm + " > net%d.log" % (num)
                            with open("run_v3.sh", "a") as f:
                                f.write(comm+'\n')
                            num += 1
