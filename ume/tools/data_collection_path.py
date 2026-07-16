import datetime
from time import strftime, localtime

def get_data_path(data_name, output_dir: str, timestamp: float) -> str:
    timestamp = float(timestamp)
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    ts_dir = f"{strftime('%Y_%m_%d_%H_%M_%S', localtime(timestamp))}_{timezone}"
    return f"{output_dir}/{ts_dir}/{data_name}"
