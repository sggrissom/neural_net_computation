NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE         = neuralNet
OBJ         = main.o support.o parse.o
ifndef $(ARG)
	ARG = 2
endif

default: $(EXE)

main.o: main.cu kernel.cu support.h dataParser.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

parse.o: dataParser.cpp dataParser.h
	g++ -c -o $@ dataParser.cpp

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

open:
	vim -O $(EXE).o* $(EXE).e*

run: default
	submitjob $(EXE) $(ARG)

rmout:
	rm -rf $(EXE).o* $(EXE).e*

clean:
	rm -rf *.o $(EXE)

cleanall: rmout clean

