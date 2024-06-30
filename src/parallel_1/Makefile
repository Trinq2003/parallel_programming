# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -pedantic -Werror
NVCCFLAGS = -O3 -arch=sm_60

# Directories
SRCDIR = .
OBJDIR = obj
HEADERDIR = header
VTKDIR = vtk

# Source files
SRC = $(SRCDIR)/helper.c $(SRCDIR)/init.c $(SRCDIR)/boundary_val.c $(SRCDIR)/uvp.c $(SRCDIR)/sor.c $(SRCDIR)/main.c $(SRCDIR)/visual.c $(SRCDIR)/cuda_kernels.cu

# Object files
OBJ = $(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
OBJ := $(OBJ:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# Compilation rules for C files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) -I$(HEADERDIR) $< -o $@

# Compilation rules for CUDA files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) -c $(NVCCFLAGS) -I$(HEADERDIR) $< -o $@

# Linking rule
all: $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o sim $(OBJ) -lm -lcuda -lcudart

# Clean up
clean:
	rm -f $(OBJDIR)/*.o sim

clean_vtk:
	rm -f $(VTKDIR)/*.vtk
