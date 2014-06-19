cimport numpy
from libc.stddef cimport ptrdiff_t
from libc.string cimport strcpy
import numpy

cdef extern from "bigfile.c":
    struct CBigFile "BigFile":
        char * basename

    struct CBigBlock "BigBlock":
        char * dtype
        int nmemb
        char * basename
        size_t size
        int Nfile

    struct CBigBlockPtr "BigBlockPtr":
        pass

    struct CBigArray "BigArray":
        int ndim
        char * dtype
        ptrdiff_t * dims
        ptrdiff_t * strides
        size_t size
        void * data

    struct CBigBlockAttr "BigBlockAttr":
        int nmemb
        char dtype[8]
        char * name
        char * data

    void big_file_set_buffer_size(size_t bytes)
    int big_block_open(CBigBlock * bb, char * basename)
    int big_block_create(CBigBlock * bb, char * basename, char * dtype, int nmemb, int Nfile, size_t fsize[])
    int big_block_close(CBigBlock * block)
    int big_block_seek(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t offset)
    int big_block_seek_rel(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t rel)
    int big_block_read(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array)
    int big_block_write(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array)
    int big_block_set_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    int big_block_get_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    CBigBlockAttr * big_block_lookup_attr(CBigBlock * block, char * attrname)
    CBigBlockAttr * big_block_list_attrs(CBigBlock * block, size_t * count)
    int big_array_init(CBigArray * array, void * buf, char * dtype, int ndim, size_t dims[], ptrdiff_t strides[])

    int big_file_open_block(CBigFile * bf, CBigBlock * block, char * blockname)
    int big_file_create_block(CBigFile * bf, CBigBlock * block, char * blockname, char * dtype, int nmemb, int Nfile, size_t fsize[])
    int big_file_open(CBigFile * bf, char * basename)
    void big_file_close(CBigFile * bf)

def set_buffer_size(bytes):
    big_file_set_buffer_size(bytes)

class BigFileError(Exception):
    pass
cdef class BigFile:
    cdef CBigFile bf
    cdef int closed

    def __cinit__(self):
        self.closed = True
    def __init__(self, filename):
        big_file_open(&self.bf, filename)
        self.closed = False

    def __dealloc__(self):
        if not self.closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        big_file_close(&self.bf)
        self.closed = True

    def open(self, block):
        cdef BigBlock rt = BigBlock()
        if 0 != big_file_open_block(&self.bf, &rt.bb, block):
            raise BigFileError("can't open block")
        rt.closed = False
        return rt

cdef class BigBlockAttrSet:
    cdef CBigBlock * bb
    property keys:
        def __get__(self):
            cdef size_t count
            cdef CBigBlockAttr * list
            list = big_block_list_attrs(self.bb, &count)
            return [list[i].name for i in range(count)]

    def __iter__(self):
        return iter(self.keys)

    def __contains__(self, name):
        cdef CBigBlockAttr * attr = big_block_lookup_attr(self.bb, name)
        if attr == NULL:
            return False
        return True

    def __getitem__(self, name):
        cdef CBigBlockAttr * attr = big_block_lookup_attr(self.bb, name)
        if attr == NULL:
            raise KeyError("attr not found")
        cdef numpy.ndarray result = numpy.empty(attr[0].nmemb, attr[0].dtype)
        if(0 != big_block_get_attr(self.bb, name, result.data, attr[0].dtype,
            attr[0].nmemb)):
            raise BigFileError("Failed to get attr")
        return result

    def __setitem__(self, name, value):
        cdef numpy.ndarray array = numpy.atleast_1d(value).ravel()

        if(0 != big_block_set_attr(self.bb, name, array.data, 
                array.dtype.base.str,
                array.shape[0])):
            raise BigFileError("Failed to save attr");

cdef class BigBlock:
    cdef CBigBlock bb
    cdef int closed

    cdef readonly BigBlockAttrSet attrs

    property size:
        def __get__(self):
            return self.bb.size
    
    property dtype:
        def __get__(self):
            return numpy.dtype((self.bb.dtype, self.bb.nmemb))

    def __cinit__(self):
        self.closed = True
        self.attrs = BigBlockAttrSet()
        self.attrs.bb = &self.bb

    def __init__(self):
#        , filename, create=False, Nfile=None, dtype=None, size=None):
#        if create:
#            self.create(filename, Nfile, dtype, size)
#        else:
#            self.open(filename)
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def open(filename):
        cdef BigBlock self = BigBlock()
        if 0 != big_block_open(&self.bb, filename):
            raise BigFileError("Failed to open file")
        self.closed = False
        return self
    @staticmethod
    def create(filename, Nfile, dtype, size):
        cdef BigBlock self = BigBlock()
        dtype = numpy.dtype(dtype)
        assert len(dtype.shape) <= 1
        if len(dtype.shape) == 0:
            items = 1
        else:
            items = dtype.shape[0]
        cdef numpy.ndarray fsize = numpy.empty(dtype='intp', shape=Nfile)
        fsize[:] = (numpy.arange(Nfile) + 1) * size / Nfile \
                 - (numpy.arange(Nfile)) * size / Nfile
        if 0 != big_block_create(&self.bb, filename, dtype.base.str,
                items, Nfile, <size_t*> fsize.data):
            raise BigFileError("Failed to create file")
        self.closed = False
        return self

    def write(self, start, numpy.ndarray buf):
        cdef CBigArray array
        cdef CBigBlockPtr ptr
        big_array_init(&array, buf.data, buf.dtype.str, 
                buf.ndim, 
                <size_t *> buf.shape,
                <ptrdiff_t *> buf.strides)
        big_block_seek(&self.bb, &ptr, start)
        big_block_write(&self.bb, &ptr, &array)

    def read(self, start, length):
        cdef numpy.ndarray result 
        cdef CBigArray array
        cdef CBigBlockPtr ptr
        cdef int i
        if length == -1:
            length = self.size - start
        if length + start > self.size:
            length = self.size - start
        result = numpy.empty(dtype=self.dtype, shape=length)
        big_array_init(&array, result.data, self.bb.dtype, 
                result.ndim, 
                <size_t *> result.shape,
                <ptrdiff_t *> result.strides)
        big_block_seek(&self.bb, &ptr, start)
        big_block_read(&self.bb, &ptr, &array)
        return result

    def close(self):
        if not self.closed:
            big_block_close(&self.bb)
            self.closed = True

    def __dealloc__(self):
        if not self.closed:
            big_block_close(&self.bb)

    def __repr__(self):
        return "<CBigBlock: %s dtype=%s, size=%d>" % (self.bb.basename,
                self.dtype, self.size)


