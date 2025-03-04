# Code Citations

## License: CC0_1_0
https://github.com/meichlseder/pyascon/tree/964242d4fd6aedac5fa938b83815ee9c31926f49/ascon.py

```
to_bytes(bytes))])

def bytes_to_state(bytes):
    return [bytes_to_int(bytes[8*w:8*(w+1)]) for w in range(5)]

def int_to_bytes(integer, nbytes):
    return to_bytes([(integer >
```


## License: unknown
https://github.com/aneeshkandi14/ascon-hw-public/tree/2cc592d0f930e31881e4821107142cb655ac4a3f/testing_codes_python/ascon.py

```
def rotr(val, r):
    return (val >> r) | ((val & (1<<r)-1) << (64-r))

def bytes_to_hex(b):
    return b.hex()
    #return "".join
```

