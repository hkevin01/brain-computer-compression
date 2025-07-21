from src.utils import data_format_handler

def test_load_nev():
    arr = data_format_handler.load_nev('dummy.nev')
    assert arr.size == 0

def test_load_nsx():
    arr = data_format_handler.load_nsx('dummy.nsx')
    assert arr.size == 0

def test_load_hdf5():
    arr = data_format_handler.load_hdf5('dummy.h5')
    assert arr.size == 0
