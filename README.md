# About
This repository contains code utilized for training and evaluating enhanced [trankit](https://github.com/nlp-uoregon/trankit) models. These models were trained on bigger datasets than those provided by trankit authors. An iteration trained on UD Slovenian-SSJ data outperformed the original trankit model over all metrics on the [SloBench leaderboard](https://slobench.cjvt.si/leaderboard/view/11). We suspect that later versions (2.15) performed slightly worse on the leaderboard, due to latest manual annotation changes in the later datasets.

For a detailed understanding of the inner workings and trankit library options, please refer to the [original documentation](https://github.com/nlp-uoregon/trankit). This repository serves as an illustration, demonstrating how to leverage the improved models developed during this project.

## Published models
| Name             | Data                                                                                                                                                                                                                                        | Model                                   |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| CJVT-Trankit-SSJ | [SSJ r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/86f832a8a0663d908fdaf5cded8c0567508fd7c0)                                                                                                                        | [zip](http://hdl.handle.net/11356/1870) | 
| CJVT-Trankit_SSJ+SST | [SSJ r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/86f832a8a0663d908fdaf5cded8c0567508fd7c0) + [SST r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/9d67eb90ae9aa6f37a7097d03d9e8864996c0609) | [zip](http://hdl.handle.net/11356/1909) |
| CJVT-Trankit_SSJ-2.15 | [SSJ r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/r2.15) + [SST r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/r2.15)                                                                       | [zip](http://hdl.handle.net/11356/1963) |
| CJVT-Trankit_SSJ+SST-2.15 | [SSJ r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/r2.15) + [SST r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/r2.15)                                                                       | [zip](http://hdl.handle.net/11356/1997) |


# Usage example
Below, we provide a step-by-step guide on how to use our models with the trankit tool.

## Step 1: Initialization
```python
from trankit import Pipeline, trankit2conllu

# Initialize trankit
p = Pipeline(lang='customized', cache_dir='<PATH TO DOWNLOADED MODELS>', embedding='xlm-roberta-large')
```

## Step 2: Process Input
There are two options for processing input:

### Option 1 - Using Text Input:
```python
text = 'Example text!'
dict_output = p(text)
```

### Option 2 - Using a Pre-tokenized List:
```python
pretokenized_list = [['Example', 'pre-tokenized', 'list', '!']]
dict_output = p(pretokenized_list)
```

## Step 3: Convert Output to CONLLu Format
```python
# Convert output from dictionary to CONLLu format
conllu_output = trankit2conllu(dict_output)
```