AM_CPPFLAGS = -I$(srcdir)/../include $(ICU_CPPFLAGS)

lib_LTLIBRARIES = libfst.la
libfst_la_SOURCES = compat.cc flags.cc fst.cc properties.cc symbol-table.cc util.cc \
                    symbol-table-ops.cc mapped-file.cc weight.cc
libfst_la_LDFLAGS = -version-info 8:0:0
libfst_la_LIBADD = $(DL_LIBS)
