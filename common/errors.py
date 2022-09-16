import datetime


class ConfigurationError(Exception):
    pass


def log(*msg, level='warn'):
    s = datetime.datetime.now().replace(microsecond=0).isoformat() + f'Z logger=dsp_block level={level} '
    s += ' '.join([str(i) for i in msg])
    print(s)
