CAFFE_ROOT="/usr/caffe"

import os
os.chdir(CAFFE_ROOT) # change the current directory to the caffe root, to help
                     # with the relative paths

import caffe
USE_GPU = True

if USE_GPU:
    caffe.set_device(0) # Or the index of the GPU you want to use
    caffe.set_mode_gpu()
    # Multi-GPU training is not available from Python, see
    # https://github.com/BVLC/caffe/issues/2936
else:
    caffe.set_mode_cpu()

print("Initialized caffe")

solver_file = "examples/mnist/lenet_solver.prototxt"

solver = caffe.SGDSolver(solver_file)
solver.solve()

snapshot_file = "examples/mnist/lenet_iter_5000.solverstate"
solver.solve(snapshot_file)