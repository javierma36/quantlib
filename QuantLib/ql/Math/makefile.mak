
# $Id$

.autodepend
.silent

# Debug version
!ifdef DEBUG
    _D = _d
!endif

# Directories
INCLUDE_DIR    = ..\..
BCC_INCLUDE    = $(MAKEDIR)\..\include

# Object files
OBJS = chisquaredistribution.obj$(_D) \
       errorfunction.obj$(_D) \
       gammadistribution.obj$(_D) \
       hstatistics.obj$(_D) \
       matrix.obj$(_D) \
       multivariateaccumulator.obj$(_D) \
       normaldistribution.obj$(_D) \
       primenumbers.obj$(_D) \
       statistics.obj$(_D) \
       symmetricschurdecomposition.obj$(_D)

# Tools to be used
CC        = bcc32
TLIB      = tlib

# Options
CC_OPTS        = -vi- -q -c -tWM -n$(OUTPUT_DIR) \
    -w-8026 -w-8027 -w-8012 \
    -I$(INCLUDE_DIR) \
    -I$(BCC_INCLUDE)
!ifdef DEBUG
CC_OPTS = $(CC_OPTS) -v -DQL_DEBUG
!endif

TLIB_OPTS    = /P128
!ifdef DEBUG
TLIB_OPTS    = /P128
!endif

# Generic rules
.cpp.obj:
    $(CC) $(CC_OPTS) $<
.cpp.obj$(_D):
    $(CC) $(CC_OPTS) -o$@ $<

# Primary target:
# static library
Math$(_D).lib:: $(OBJS)
    if exist Math$(_D).lib     del Math$(_D).lib
    $(TLIB) $(TLIB_OPTS) Math$(_D).lib /a $(OBJS)


# Clean up
clean::
    if exist *.obj         del /q *.obj
    if exist *.obj$(_D)    del /q *.obj
    if exist *.lib   del /q *.lib

