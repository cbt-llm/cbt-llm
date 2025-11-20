# cbt-llm
CMSC691: Reward Based - Action Oriented CBT LLMs

## Project Setup

```
git clone https://github.com/cbt-llm/cbt-llm.git
cd cbt-llm
cp .env.example .env
```

### Add PyMedTermino-0.3.2
 
1. Under ```external_libs``` add the PyMedTermino-0.3.2 folder

2. Update ```external_libs/PyMedTermino-0.3.2/setup.py``` with the following:
```
import os
SNOMEDCT_DIR = os.getenv("SNOMEDCT_DIR")
SNOMEDCT_CORE_FILE = os.getenv("SNOMEDCT_CORE_FILE")
```



### Add SNOMED CT Data

PyMedTermino requires access to the official SNOMED CT distribution, which is protected by a copyright license.
You can obtain the license and then upload the release files into 

```data/external/snomed```

Accordingly, add the relative path to your ```.env```

It should look like:

```
SNOMEDCT_DIR = "data/external/snomed/SNOMEDCT_CORE_SUBSET_201611"
SNOMEDCT_CORE_FILE = "data/external/snomed/SnomedCT_RF2Release_INT_20160731/SNOMEDCT_CORE_SUBSET_201611.txt"
```

These paths point to the directory and core file that PyMedTermino will use. Then continue as follows:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install external_libs/PyMedTermino-0.3.2
```

## Graph Setup

### Load From PymedTermino
```python -m cbt_llm.pymed_loader```

### Load SNOMED Concepts/Relationships to neo4j Graph
```python -m cbt_llm.pymed_graph```

