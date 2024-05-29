# Raven Home Task



The task is in Analysis.ipynb.<br>
Run it with docker or see the Analysis.html file for static report.


### Run With Docker

```python
docker build -t raven-image .
```

```python
docker run -it --rm -p 8888:8888 -v $(pwd):/raven_wd raven-image  # run with jupyter notebook
```

<br>

### Project Structure

```
├── Analysis.ipynb  # my solution
├── Dockerfile
├── Home Assignment - data scientist May24.pdf
├── README.md
├── requirements.txt
├── src  # project source
│   ├── df.parquet.gzip
│   ├── feature_engineering.py
│   ├── __init__.py
│   ├── model.py
│   ├── preprocessor.py
│   └── utils.py
├── static  # constants
│   ├── constants.py
└── tests  # unittesting
    └── test_logic.py
```

