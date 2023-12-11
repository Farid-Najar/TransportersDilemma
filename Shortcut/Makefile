DIR=/opt/homebrew/opt/python@3.10/Frameworks/Python.framework/Versions/3.10/include/python3.10
CC=gcc
CFLAGS=-I$(DIR)
ODIR=.

LIBS_DIR=/opt/homebrew/opt/python@3.10/Frameworks/Python.framework/Versions/3.10/include/python3.10
NUMPY=/opt/homebrew/lib/python3.10/site-packages/numpy/core/include
# LIBS=-lpython3.10

_DEPS =
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = package.so
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.so: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
#$(CC) -c -o $@ $< $(CFLAGS)

shortcut: $(OBJ)
	gcc -shared $^ $(CFLAGS) -I$(LIBS_DIR) -I$(NUMPY) -o $@ 
#$(LIBS)

# .PHONY: clean
# clean:
#         rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~