# Multilabel Image Classification

## Setup python dependencies

```bash
$ python -m venv .venv

# activate venv and install additional dependencies
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Train model
```bash
# -m: path to save the best model
# -i: path to images folder
# -l: path to label csv file
# -e: number of epochs
# -r: learning rate
$ python train.py -m "../models/bag_detect
ion_v3.pth" -i "../data/images" -l "../data/labeledDatapoints.csv" -e 50 -r 0.005
```

The trained best model is checked in at `models/bag_detection_v3.pth`, which yielded the best weighted f1: {'train': 0.9551, 'val': 0.9187} 

After the training done, you can see the metrics from MLFlow dashboard
```bash
python -m mlflow ui --port 5000
# and open http://localhost:5000
```

## Run model server
```bash
$ python app.py
```

## Call model endpoint
```python
res = requests.post(
    "http://localhost:5000/predict",
    data=image_byte_array,
)
res.json()
# result json:
# {
#     "score": model score,
#     "label": class name,
# }
```