import redis
import numpy as np
import struct
from typing import Dict, List, Literal, Optional, Union

REDIS_ENCODERS = {
    str: lambda x: dict(data=x.encode()),
    float: lambda x: dict(data=struct.pack("d", x)),
    np.ndarray: lambda x: dict(data=x.tobytes(), shape=str(x.shape), dtype=str(x.dtype)),
}

def np_array_decoder(data: bytes, shape: str, dtype: str) -> np.ndarray:
    shape_tuple = tuple(int(dim) for dim in shape.strip("()").split(",") if dim)
    return np.frombuffer(data, dtype=dtype).reshape(shape_tuple)

REDIS_DECODERS = {
    str: lambda d: d[b"data"].decode(),
    float: lambda d: struct.unpack("d", d[b"data"])[0],
    np.ndarray: lambda d: np_array_decoder(d[b"data"], d[b"shape"].decode(), d[b"dtype"].decode())
}

class RedisClient:
    
    def __init__(self, host, port):
        self.redis_client = redis.Redis(host=host, port=port)

    def stream_add_batch(self, batch: Dict[str, np.ndarray], batch_maxlen: Dict[str, int] = {}):
        with self.redis_client.pipeline() as pipe:
            for stream_key, stream_value in batch.items():
                stream_value_encoded = REDIS_ENCODERS[type(stream_value)](stream_value)
                pipe.xadd(stream_key, stream_value_encoded, maxlen=batch_maxlen.get(stream_key))
            res = pipe.execute()
        return res

    def stream_get_batch(self, stream_keys: Dict[str, type]):
        with self.redis_client.pipeline() as pipe:
            for stream_key in stream_keys:
                pipe.xrevrange(stream_key, count=1)
            results = pipe.execute()
        res = dict()
        for (stream_key, value_type), entries in zip(stream_keys.items(), results):
            if len(entries) == 0:
                res[stream_key] = None
            else:
                _, entry_data = entries[0]
                decoded_value = REDIS_DECODERS[value_type](entry_data)
                res[stream_key] = decoded_value
        return res

    def stream_get_len(self, stream_key: str) -> int:
        return self.redis_client.xlen(stream_key)

    def stream_get_batch_after(self, stream_keys: Dict[str, type], timestamps: Dict[str, float], block: int = None, out_as_np_array: bool = True):
        timestamps_int_ms = {k: int(v * 1000) for k, v in timestamps.items()}
        streams_filter = {k: f"{timestamps_int_ms[k]}-0" for k in stream_keys}
        with self.redis_client.pipeline() as pipe:
            raw_results = self.redis_client.xread(streams=streams_filter, count=None, block=block)
            output_insertion_timestamps, output_data = dict(), dict()
            for stream_key, list_of_entries in raw_results:
                stream_key = stream_key.decode()
                output_insertion_timestamps[stream_key] = list()
                output_data[stream_key] = list()
                for entry in list_of_entries:
                    entry_id, entry_data = entry
                    output_insertion_timestamps[stream_key].append(int(entry_id.decode().split("-")[0]) / 1000.0)
                    output_data[stream_key].append(REDIS_DECODERS[stream_keys[stream_key]](entry_data))
            if out_as_np_array:
                for stream_key in output_data:
                    output_data[stream_key] = np.array(output_data[stream_key])
            return output_insertion_timestamps, output_data
