# Deep city noise
A deep learning project for urban sound classification using the benchmark dataset UrbanSound8K. 
The goal is to understand which feature extraction method is most effective when training a Convolutional Neural Networks (CNNs).

## Replicate the experiment

Install Python 3.11.11. Execute the following command.

```bash
git clone https://github.com/aledigirm3/deep-city-noise.git
cd deep-city-noise
pip install -r requirements.txt
```

Before running the code, it is important to download the dataset from the following link.

🔗 https://urbansounddataset.weebly.com/urbansound8k.html

After downloading the dataset as a zip file, extract the 'UrbanSound8K' folder and place it in the current directory.

Now move to the 'src' folder
```bash
  cd src
```

Select type of feature extraction in 'config.py' file.
for example:
```bash
  FEATURE_TYPE = 'mel'
```
you can visualyze the feature extractors; see the file jupyter 'data_analysis.ipynb

Now you can start the training, which will provide you with the results of the chosen method across the various folds.
```bash
  python train.py
```
The system is CUDA-enabled, If available on the system.

### 📄 If you want to explore the topic further, including the approaches used and the system pipeline, refer to the PDF available in the repository. 📄

