import re
from unidecode import unidecode


def convert_to_ascii(string):
    return unidecode(string)


def collapse_whitespace(string):
    return re.sub(_whitespace_re, ' ', string)


def put_spacing_num(string):
    string = re.sub('[A-Za-z]+', lambda ele: ' ' + ele[0] + ' ', string)
    return re.sub(r'[ ]+', ' ', string).strip()
