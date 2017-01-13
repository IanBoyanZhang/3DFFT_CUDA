#
# Makefile for building Matrix Multiplication under CUDA
# 

# HOST = $(shell hostname)
LILLIPUT = $(shell hostname | grep lilliput | wc -c)
CSECLASS = $(shell hostname | grep cseclass | wc -c)
CARVER = $(shell hostname | grep cvrsvc | wc -c)
MACS = $(shell hostname | grep apm2402 | wc -c)


ifneq ($(MACS), 0)
PUB = /share/class/public/cse260-fa12
include $(PUB)/Arch/arch.cuda.mac
else
ifneq ($(LILLIPUT), 0)
PUB     = /class/public/cse260-fa12
include $(PUB)/Arch/arch.cuda.gnu
else
ifneq ($(CSECLASS), 0)
PUB	= /class/public/cse260-fa12
include $(PUB)/Arch/arch.cuda.gnu
else
ifneq ($(CARVER), 0)
PUB = /global/u2/b/baden/cse260-fa12
include $(PUB)/Arch/arch.cuda.gnu
endif
endif
endif
endif

# Set usecache=1 to build the variant that uses cache
# By default, build the code to use shared memory
ifeq ($(usecache), 1)
    C++FLAGS += -DUSE_CACHE
    CFLAGS += -DUSE_CACHE
    NVCCFLAGS += -DUSE_CACHE
endif

# Set debug=1 on make command line to keep symbol table info for gdb/cachegrind
ifeq ($(debug), 1)
        CFLAGS += -g
        NVCCFLAGS += -g
        LDFLAGS += -g
        C++FLAGS += -g
endif   


# Set gprof=1 on make command line to compile for gprof profiler
ifeq ($(gprof), 1)
        CFLAGS += -g -pg
        NVCCFLAGS += -g -pg
        C++FLAGS += -g -pg
        LDFLAGS += -g -pg
endif

# If you want to compile for single precision,
# specify single=1 on the "make" command line
ifeq ($(single), 1)
else
    C++FLAGS += -D_DOUBLE
    CFLAGS += -D_DOUBLE
    NVCCFLAGS += -D_DOUBLE
endif

# Keep arround compiler output files, including the ptx assembler
ifeq ($(keep), 1)
	NVCCFLAGS	+= -keep
	NVCCFLAGS 	+= --ptx
endif

# If you want to use the  CUDA Timer
# specify cuda_timer=1 on the "make" command line
# NVCCFLAGS += -DCUDA_TIMER
ifeq ($(cuda_timer), 1)
	NVCCFLAGS += -DCUDA_TIMER
endif

# Uncomment if you want to report resource requirements (registers etc)
NVCCFLAGS += --ptxas-options=-v
# NVCCFLAGS += --opencc-options -LIST:source=on

# You can set the thread block geometry by specifying bx= and by= on
# the make command line, e.g. make bx=16 by=32
# This feature is useful for the shared memory variant but
# not for the naive variant

ifdef bx
	DIM_X = -DBLOCKDIM_X=$(bx)
endif
ifdef by
	DIM_Y = -DBLOCKDIM_Y=$(by)
endif

BLOCKING = $(DIM_X) $(DIM_Y)
NVCCFLAGS += $(BLOCKING)
CFLAGS += $(BLOCKING)
C++FLAGS += $(BLOCKING)

APP=fft

OBJECTS = main.o util.o

$(APP): $(OBJECTS) fft.o
	$(C++) -fPIC -o $@ $(LDFLAGS) $(OBJECTS)  $(LDLIBS)

clean:
	rm -f *.linkinfo *.o  *.vcproj $(APP)

