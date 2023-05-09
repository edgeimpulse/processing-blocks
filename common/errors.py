import datetime
import json
import traceback


class ConfigurationError(Exception):
    pass


def log(*msg, level='warn'):
    msg_clean = ' '.join([str(i) for i in msg])
    print(json.dumps(
        {'msg': msg_clean,
         'level': level,
         'time': datetime.datetime.now().replace(microsecond=0).isoformat() + 'Z'}))


def log_exception(msg):
    log(msg + ': ' + traceback.format_exc(), level='error')
