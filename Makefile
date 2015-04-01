ROOT_DIR_CCP := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/cpp_source
ROOT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
INCLUDE := -I /usr/local/cuda/include -I $(ROOT_DIR)/cpp_source 
LIB := -L /usr/local/cuda/lib64 -lcudart -lcuda -lcurand
FILES := $(ROOT_DIR_CCP)/basics.cu $(ROOT_DIR_CCP)/Tensor.cu $(ROOT_DIR_CCP)/pythonInterface.cu $(ROOT_DIR_CCP)/cudaKernels.cu  $(ROOT_DIR_CCP)/gpupy.cpp
COMPUTE_CAPABILITY := arch=compute_35,code=sm_35 

all:	
	#nvcc $(FILES) $(INCLUDE) $(LIB) -o test.out
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB)
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink gpupy.o basics.o Tensor.o pythonInterface.o cudaKernels.o -o link.o 
	g++ -shared -o $(ROOT_DIR)/py_source/gpupylib.so gpupy.o basics.o Tensor.o pythonInterface.o cudaKernels.o link.o $(LIB) $(INCLUDE)

clean:
	rm *.o *.out $(ROOT_DIR)/py_source/gpupylib.so
