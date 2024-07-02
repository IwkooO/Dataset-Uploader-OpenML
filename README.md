# Dataset-Uploader-OpenML
This repository is an automated dataset uploader for OpenML


## Installation
Recommended Python version: 3.10.4
requirements.txt contains all the necessary packages to run the code. You can install them by running the following command:

```
pip install -r requirements.txt
```

## Usage
To run the code, you need to have an OpenML account. You can create one [here](https://www.openml.org/). After creating an account, you need to get your API key from your account settings.

To run the code, you need to have a working OpenAI API key. This key should be inserted in .env file or you can directly insert it in the code in functions.py file.


You can run the code by running the following command:

```
python whole_app.py
```



## Meaning of the files
- whole_app.py: This is the main file that you need to run to open the uploader.
- feature_detection.py : This file contains the functions that are used to detect the features of the dataset (sortinghatinf).
- functions.py : This file contains the functions used in the uploader.



