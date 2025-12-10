# About
This repository contains code utilized for retraining and evaluating models based on [Trankit: A Light-Weight Transformer-based Python Toolkit for Multilingual Natural Language Processing](https://github.com/nlp-uoregon/trankit). Using this setup, we developed new Trankit models for Slovenian, trained on a more recent and considerably larger version of the Slovenian UD Treebanks than the default Trankit models (trained on UD v2.5). 

For a detailed understanding of the inner workings and Trankit library options, please refer to the [original documentation](https://github.com/nlp-uoregon/trankit). This repository serves as an illustration, demonstrating how to leverage the improved models developed during this project.

## Published models

The models were trained on the successive versions of the SSJ UD treebank of written Slovenian, the SST UD treebank of spoken Slovenian, and a combined dataset incorporating both. 

For production use, we recommend the latest model, [Trankit SSJ+SST-2.15](http://hdl.handle.net/11356/1997), which achieves state-of-the-art [performance](#performance) for both written and spoken Slovenian.

| Release date              | Short name             | Training Data                                                                                                                                                                                                                                        | Model (CLARIN.SI repository)                                   |
|------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| 2023-09-29 | Trankit-SSJ-2.12 | [SSJ r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/86f832a8a0663d908fdaf5cded8c0567508fd7c0)                                                                                                                        | [zip](http://hdl.handle.net/11356/1870) | 
| 2024-01-17 | Trankit_SSJ+SST-2.12 | [SSJ r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/86f832a8a0663d908fdaf5cded8c0567508fd7c0) + [SST r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/9d67eb90ae9aa6f37a7097d03d9e8864996c0609) | [zip](http://hdl.handle.net/11356/1909) |
| 2024-08-29 | Trankit_SSJ-2.14 | [SSJ r2.14](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/r2.14)                                                                       | [zip](http://hdl.handle.net/11356/1963) |
| 2024-12-06 | Trankit_SST-2.15 | [SST r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/r2.15)                                                                       | [zip](http://hdl.handle.net/11356/1996) |
| 2024-12-06 | Trankit_SSJ+SST-2.15 | [SSJ r2.14](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/r2.14) + [SST r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/r2.15)                                                                       | [zip](http://hdl.handle.net/11356/1997) **--> recommended**|




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

# Performance
<p>
The table below reports lemmatization (Lemmas), tagging (UPOS), full morphological analysis (XPOS) 
and parsing (LAS) performance on the written <b>SSJ-2.12</b> test set and the spoken 
<b>SST-2.15</b> test set.
</p>

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Model type</th>
      <th colspan="4">SSJ-2.12-test (written)</th>
      <th colspan="4">SST-2.15-test (spoken)</th>
    </tr>
    <tr>
      <th>Lemmas</th><th>UPOS</th><th>XPOS</th><th>LAS</th>
      <th>Lemmas</th><th>UPOS</th><th>XPOS</th><th>LAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Trankit-SSJ-2.12</td><td>Written</td>
      <td>98.07</td><td>99.12</td><td>98.24</td><td>95.48</td>
      <td>98.16</td><td>95.33</td><td>93.93</td><td>79.14</td>
    </tr>
    <tr>
      <td>Trankit-SST-2.15</td><td>Spoken</td>
      <td>94.27</td><td>97.74</td><td>93.74</td><td>91.90</td>
      <td>97.90</td><td>98.79</td><td>96.71</td><td>86.54</td>
    </tr>
    <tr style="font-weight: bold;">
      <td><b>Trankit-SSJ+SST-2.15</b></td><td><b>Written+Spoken</b></td>
      <td><b>98.10</b></td><td><b>99.17</b></td><td><b>98.27</b></td><td><b>95.36</b></td>
      <td><b>98.85</b></td><td><b>98.97</b></td><td><b>98.02</b></td><td><b>87.93</b></td>
    </tr>
  </tbody>
</table>


# Acknowledgement
This work was supported by Slovenian Research and Innovation Agency through research project [SPOT: A Treebank-Driven Approach to the Study of Spoken Slovenian (Z6-4617)](https://spot.ff.uni-lj.si/) and research programme Language Resources and Technologies for Slovene (P6-0411). Infrastructural support was provided by the Centre for Language Resources and Technologies at the University of Ljubljana ([CJVT](https://www.cjvt.si/en/)).
