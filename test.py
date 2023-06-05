import json
import sys
import numpy as np
data = {
    "name": [0,1,0,1,0,1,0,1]*10
}

json_data = json.dumps(data)
byte_size = sys.getsizeof(json_data)

print("Size of JSON object in bytes:", byte_size)

