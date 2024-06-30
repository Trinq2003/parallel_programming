CC = gcc
CFLAGS = -Wall -pedantic -Werror
OBJDIR = obj
SRCDIR = .
HEADERDIR = header
VTKDIR = vtk

SRC = $(SRCDIR)/helper.c $(SRCDIR)/init.c $(SRCDIR)/boundary_val.c $(SRCDIR)/uvp.c $(SRCDIR)/sor.c $(SRCDIR)/main.c $(SRCDIR)/visual.c

OBJ = $(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) $< -o $@

all: $(OBJ)
	$(CC) $(CFLAGS) -o sim $(OBJ) -lm

clean:
	rm -f $(OBJDIR)/*.o sim

clean_vtk:
	rm -f $(VTKDIR)/*.vtk
