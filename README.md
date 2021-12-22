# MLSecurityProject
this is the final project code for MLS 2021
## For the first method:
Since the second method has better performance, the eval program of the first method would only return the accuracy and asr on the validation data, does not support the predict for a single image.
usage:
```shell script 
python mask.py model_filename clean_data_filename poison_data_filename
# output: clean accuracy and asr
```

example: 
```shell script
python mask.py models/bd_net.h5 data/cl/valid.h5 data/bd/valid.h5
# output: 
# Clean Classification accuracy: 79.44054732831039
# Attack Success Rate: 8.539014462630986

```
## For the second method:
usage:
```shell script 
python p_eval.py model_filename data_filename img_filename
# output: a number
```

example: 
```shell script
python p_eval.py models/bd_net.h5 data/valid.h5 img.png
# output: 1283
```