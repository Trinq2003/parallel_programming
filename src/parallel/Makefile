CC = gcc
NVCC = nvcc
CFLAGS = -Wall -pedantic -Werror
LDFLAGS = -lm
OBJDIR = obj
SRCDIR = .
HEADERDIR = header
CUDADIR = cuda
VTKDIR = vtk

SRC = $(SRCDIR)/helper.c $(SRCDIR)/init.c $(SRCDIR)/boundary_val.c $(SRCDIR)/uvp.c $(SRCDIR)/sor.c $(SRCDIR)/main.c $(SRCDIR)/visual.c
CUDA_SRC = $(CUDADIR)/cuda_kernels.cu

OBJ = $(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
CUDA_OBJ = $(CUDA_SRC:$(CUDADIR)/%.cu=$(OBJDIR)/%.o)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) $< -o $@

$(OBJDIR)/%.o: $(CUDADIR)/%.cu
	$(NVCC) -c $< -o $@

all: $(OBJDIR) $(OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o sim $(OBJ) $(CUDA_OBJ) $(LDFLAGS) -lcudart -L/usr/local/cuda/lib64

clean:
	rm -f $(OBJDIR)/*.o sim

clean_vtk:
	rm -f $(VTKDIR)/*.vtk