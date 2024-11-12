# /bin/bash

if ! test -f ./gs_1p/nae.json;
then
    echo "Can not find the ./gs_1p/nae.json file. Please make sure the constructed data is placed in the ./gs_1p/ folder."
fi

if ! [[ -f nr_models.py  && -f nr4at.py ]];
then
    echo "Can not find the nr_model.py file. Please make sure this support Python file is placed in the current folder."
fi

python ./nr4at.py

# remapping the output of inference back to the original port strings.
python -m graphstorm.gconstruct.remap_result \
          --node-id-mapping ./gs_1p/raw_id_mappings/ \
          --pred-ntypes airport \
          --prediction-dir ./infer/predictions/
