nohup: ignoring input
2019-04-06 09:52:02.866055: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2019-04-06 09:52:02.873075: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394370000 Hz
2019-04-06 09:52:02.873222: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x56361b280af0 executing computations on platform Host. Devices:
2019-04-06 09:52:02.873236: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
WARNING:tensorflow:From /root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/training/queue_runner_impl.py:391: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From /root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
WARNING:tensorflow:From /root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/ops/control_flow_ops.py:3632: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/ops/image_ops_impl.py:1241: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
Traceback (most recent call last):
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1334, in _do_call
    return fn(*args)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1319, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1407, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.InvalidArgumentError: You must feed a value for placeholder tensor 'Placeholder' with dtype float
	 [[{{node Placeholder}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "get_hashcode_mini.py", line 94, in <module>
    main()
  File "get_hashcode_mini.py", line 70, in main
    ret = sess.run(y_conv, feed_dict={x: image})
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 929, in run
    run_metadata_ptr)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1152, in _run
    feed_dict_tensor, options, run_metadata)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1328, in _do_run
    run_metadata)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1348, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: You must feed a value for placeholder tensor 'Placeholder' with dtype float
	 [[node Placeholder (defined at get_hashcode_mini.py:58) ]]

Caused by op 'Placeholder', defined at:
  File "get_hashcode_mini.py", line 94, in <module>
    main()
  File "get_hashcode_mini.py", line 58, in main
    saver = tf.train.import_meta_graph(model_dir+'model.meta')
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 1435, in import_meta_graph
    meta_graph_or_file, clear_devices, import_scope, **kwargs)[0]
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 1457, in _import_meta_graph_with_return_elements
    **kwargs))
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/meta_graph.py", line 806, in import_scoped_meta_graph_with_return_elements
    return_elements=return_elements)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/importer.py", line 442, in import_graph_def
    _ProcessNewOps(graph)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/importer.py", line 235, in _ProcessNewOps
    for new_op in graph._add_new_tf_operations(compute_devices=False):  # pylint: disable=protected-access
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 3433, in _add_new_tf_operations
    for c_op in c_api_util.new_tf_operations(self)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 3433, in <listcomp>
    for c_op in c_api_util.new_tf_operations(self)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 3325, in _create_op_from_tf_operation
    ret = Operation(c_op, self)
  File "/root/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1801, in __init__
    self._traceback = tf_stack.extract_stack()

InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'Placeholder' with dtype float
	 [[node Placeholder (defined at get_hashcode_mini.py:58) ]]

