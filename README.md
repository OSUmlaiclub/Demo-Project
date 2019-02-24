# Demo Projects


## List of Projects

* Digit Recognizer (Computer Vision Problem)
* Titanic (Data Analysis Problem)
* House-Prices (Data Analysis Problem)

## Start

### Set Up the Environment

#### Installing Applications

* [Python 3.X](https://www.python.org/downloads/)
* [miniconda](https://docs.conda.io/en/latest/miniconda.html)
* [kaggle](https://github.com/Kaggle/kaggle-api)

#### Installing Packages

* [Jupyter Notebook](https://jupyter.org/): A web-based notebook environment for interactive computing.

```
conda install -c anaconda ipython
conda install -c anaconda jupyter
```

* [NumPy](https://matplotlib.org/): Adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

```
conda install -c anaconda numpy
```

* [matplotlib](https://matplotlib.org/): Provides both a very quick way to visualize data from Python and publication-quality figures in many formats.

```
conda install -c conda-forge matplotlib
```

* [seaborn](https://seaborn.pydata.org/): Data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

```
conda install -c anaconda seaborn 
```

* [pandas](https://pandas.pydata.org/): Offers data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.

```
conda install -c anaconda pandas
```

* [scikit-learn](https://scikit-learn.org/stable/): Machine Learning Library 

```
conda install -c anaconda scikit-learn 
```

* [TensorFlow](https://www.tensorflow.org/): Machine Learning Framework (Back-end)

```
conda install -c conda-forge tensorflow
```

If you have an NVIDIA GPU with CUDA, install TensorFlow GPU also

```
# To check whether you have CUDA installed, use this command:
# nvidia-smi

conda install -c anaconda tensorflow-gpu
```

* [Keras](https://keras.io/): Machine Learning Framework, easy syntax (Front-end, on top of TensorFlow)

```
conda install -c conda-forge keras
```

#### Set up Kaggle to Download Datasets and Upload Submission

* Create an account on [Kaggle](https://www.kaggle.com/)
* Access this webpage [https://www.kaggle.com/<username>/account](https://www.kaggle.com/<username>/account), replace <username> with your Kaggle Username
* Click on **Create New API Token**, which will download a *kaggle.json* file.
* Move the downloaded file, *kaggle.json*, from Downloads to:
    * Windows: ```C:\Users\<Windows-username>\.kaggle\kaggle.json```
    * Mac/Linux: ```~/.kaggle/kaggle.json```
* Change permission of the file
    * Windows: ```chmod 600 C:\Users\<Windows-username>\.kaggle\kaggle.json```
    * Mac/Linux: ```chmod 600 ~/.kaggle/kaggle.json```

