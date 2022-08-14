pyworld_exist = True
try:
    import pyworld as pw
except Exception as e:
    pyworld_exist = False
    pw = None

pysptk_exist = True
try:
    from pysptk import sptk
except Exception as e:
    pysptk_exist = False
    sptk = None
