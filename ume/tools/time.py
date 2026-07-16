import datetime
from time import strftime, localtime

def get_time_hmstz(timestamp: float) -> str:
    " get time in h:m:s timezone format "
    timestamp = float(timestamp)
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    return f"{strftime('%H:%M:%S', localtime(timestamp))} {timezone}"

def get_datetime_str(timestamp: float) -> str:
    timestamp = float(timestamp)
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    ts_dir = f"{strftime('%Y_%m_%d_%H_%M_%S', localtime(timestamp))}_{timezone}"
    return ts_dir
