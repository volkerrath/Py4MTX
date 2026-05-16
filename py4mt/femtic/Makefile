CXX           = mpiicpx
CC            = mpiicpx  #oneapi 2025 
GCC           = mpigxx




HOST := $(shell hostname -s)
DEST	      = $(HOME)/bin
LDFLAGS       =
LIBS          =  -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -liomp5 -lpthread -lm -ldl

# MODE = debug
MODE = release
ifeq ($(MODE),release)
	# Default release option
	CXXFLAGS      = -O3 \
			-qopenmp \
			-D_LINUX \
			-DMKL_ILP64 \
			-D_USE_OMP \
			-DNDEBUG \
			-diag-disable=10441
	PROG  = femtic-dabic_152
else
	# debug option (added -g, changed to -O0)
	CXXFLAGS      = -g \
			-O0 \
			-fno-omit-frame-pointer -Wall -Wextra -Wpedantic \
			-debug inline-debug-info \
			-qopenmp \
			-D_LINUX \
			-DMKL_ILP64 \
			-D_USE_OMP \
			-diag-disable=10441
	PROG  = femtic-dabic_152
endif

HDF5_OUTPUT = yes
ifeq ($(HDF5_OUTPUT),yes)
	HDF5_ROOT=$(HOME)/local/intel/
	HDF5_STRING= hdf5
else
	HDF5_STRING=
endif


# ---------------------------------------------------------------------------
# Optional HDF5 output of sensitivity matrix / residual / error vectors.
# Set HDF5_OUTPUT = yes to enable, or leave as 'no' (default) to build
# without any HDF5 dependency.
#
# Usage:
#   make HDF5_OUTPUT=yes                        # auto-detect (see below)
#   make HDF5_OUTPUT=yes HDF5_ROOT=/path/to/hdf5  # explicit override
#
# Auto-detection order (first match wins, no pkg-config required):
#   1. HDF5_ROOT   set explicitly on the command line or in environment
#   2. HDF5_DIR    common on Cray / NERSC systems
#   3. HDF5HOME    used on some MPI-IO stacks
#   4. EBROOTHDF5  EasyBuild / Lmod convention (e.g. ICHEC Kay, DKRZ Levante)
#   5. pkg-config  fallback for workstations / conda environments
#
# On HPC clusters, 'module load HDF5' (or similar) sets one of the above
# variables automatically — no pkg-config needed.
# ---------------------------------------------------------------------------
HDF5_OUTPUT  ?= no

ifeq ($(HDF5_OUTPUT),yes)
    CXXFLAGS     += -D_WRITE_INVERSION_DATA_HDF5

    # Resolve HDF5 root from environment if not given on command line
    ifndef HDF5_ROOT
        ifdef HDF5_DIR
            HDF5_ROOT := $(HDF5_DIR)
        else ifdef HDF5HOME
            HDF5_ROOT := $(HDF5HOME)
        else ifdef EBROOTHDF5
            HDF5_ROOT := $(EBROOTHDF5)
        endif
    endif

    ifdef HDF5_ROOT
        CXXFLAGS += -I$(HDF5_ROOT)/include
        LIBS     += -L$(HDF5_ROOT)/lib -lhdf5
    else
        # Last resort: pkg-config (workstations, conda)
        CXXFLAGS += $(shell pkg-config --cflags hdf5 2>/dev/null)
        LIBS     += $(shell pkg-config --libs   hdf5 2>/dev/null)
    endif

    HDF5_OBJS    = InversionHDF5Writer.o
else
    HDF5_OBJS    =
endif


PROGRAM = $(HOST)_$(PROG)_$(HDF5_STRING)_$(MODE).x


OBJS          = AnalysisControl.o \
                ComplexSparseMatrix.o \
                ComplexSparseSquareMatrix.o \
                ComplexSparseSquareSymmetricMatrix.o \
                ComplexSparseSquareUnsymmetricMatrix.o \
                DoubleSparseMatrix.o \
                DoubleSparseSquareMatrix.o \
                DoubleSparseSquareSymmetricMatrix.o \
                DoubleSparseSquareUnsymmetricMatrix.o \
                Forward2D.o \
                Forward2DQuadrilateralElement.o \
                Forward2DNonConformingQuadrilateralElement0thOrderEdgeBased.o \
                Forward2DQuadrilateralElementEdgeBased.o \
                Forward2DSquareElement.o \
                Forward2DSquareElement0thOrderEdgeBased.o \
                Forward2DSquareElement1stOrderEdgeBased.o \
                Forward2DSquareElement1stOrderNodeBased.o \
                Forward2DSquareElement2ndOrderNodeBased.o \
                Forward2DSquareElementEdgeBased.o \
                Forward2DSquareElementNodeBased.o \
                Forward2DTriangleElement.o \
                Forward2DTriangleElement0thOrderEdgeBased.o \
                Forward2DTriangleElement1stOrderNodeBased.o \
                Forward2DTriangleElementEdgeBased.o \
                Forward2DTriangleElementNodeBased.o \
                Forward3D.o \
                Forward3DBrickElement0thOrder.o \
                Forward3DNonConformingHexaElement0thOrder.o \
                Forward3DTetraElement0thOrder.o \
                Inversion.o \
                InversionGaussNewtonModelSpace.o \
		InversionGaussNewtonDataSpace.o \
		InversionGaussNewtonDataSpace_ABIC.o \
		ConstrainingModel.o \
		main.o \
                MeshData.o \
                MeshDataBrickElement.o \
                MeshDataNonConformingHexaElement.o \
                MeshDataTetraElement.o \
                AdditinalOutputPoint.o \
                ObservedDataStation.o \
                ObservedDataStationPoint.o \
                ObservedDataStationMT.o \
                ObservedDataStationApparentResistivityAndPhase.o \
                ObservedDataStationHTF.o \
                ObservedDataStationVTF.o \
                ObservedDataStationPT.o \
                ObservedDataStationNMT.o \
                ObservedDataStationNMT2.o \
                ObservedDataStationNMT2ApparentResistivityAndPhase.o \
                ObservedData.o \
                OutputFiles.o \
                PARDISOSolver.o \
                PARDISOSolverComplex.o \
                PARDISOSolverDouble.o \
                ResistivityBlock.o \
                RougheningMatrix.o \
                RougheningSquareMatrix.o \
                Util.o \
                $(HDF5_OBJS)


all:            $(PROGRAM)

install:	$(PROGRAM)
	install -m 755 $(PROGRAM) $(DEST)
# default: release mode
$(PROGRAM):     $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(PROGRAM)

# debug: CXXFLAGS = $(DEBUG_CXXFLAGS)
# debug: $(PROGRAM)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *~ $(PROGRAM)
