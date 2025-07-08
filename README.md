Benchmarking Foundation Models with Multimodal Public Electronic Health Records
=========================
## Table of contents
* [Introduction](#introduction)
* [Structure](#structure)
* [Data Requirements](#data-requirements)
* [Workflow](#workflow)

## Introduction
Foundation models have shown great promise in processing electronic health records (EHRs), offering the flexibility to handle diverse medical data modalities such as text, time series, and images. This repository presents a comprehensive benchmark framework designed to evaluate the predictive performance, fairness, and interpretability of foundation models—both as unimodal encoders and multimodal learners—using the publicly available MIMIC-IV database.

To support consistent and reproducible evaluation, we developed a standardized data processing pipeline that harmonizes heterogeneous clin-
ical records into an analysis-ready format. We systematically compared eight foundation models, encompassing both unimodal and multimodal models, as well as domain-specific and general-purpose variants.

## Structure
The structure of this repository is detailed as follows:

- `scripts/1_...` contains the scripts for data processing pipeline
- `scripts/2_... & 3_...` contains the scripts for evaluating foundation models as unimodal encoders
- `scripts/4_..._...` contains the scripts for evaluating foundation models as multimodal learners

## Data Requirements

For data access and description, please visit: [https://mimic.mit.edu](https://mimic.mit.edu/)

MIMIC-IV v2.2 [https://physionet.org/content/mimiciv/2.2](https://physionet.org/content/mimiciv/2.2/#files-panel)

MIMIC-CXR [https://physionet.org/content/mimic-cxr/2.0.0](https://physionet.org/content/mimic-cxr/2.0.0/#files-panel)

MIMIC-IV-Note [https://physionet.org/content/mimic-iv-note/2.2/note](https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel)

## Workflow
The following sub-sections describe the workflow for our benchmark and how they should ideally be run.

### 1. Data Processing Pipeline

<div class="figure" style="text-align: center">

<img src="figures/figure1A.png" width="85%"/>

</div>

For generating master dataset, run `1_1 Master Dataset Generation.ipynb`

For generating benchmark dataset, we used
~~~
python "1_2 Data Processing Pipeline.py --input-path {input_path} --output-pkl-path {output_pkl_path} --output-csv-path {output_csv_path} --age-lower 18 --start-diff 0 --end-diff 24
~~~

**Key Arguements**:

- `input_path` : Path to master dataset.
- `output_pkl_path ` : Output path for processed ICU stay data.
- `output_csv_path ` : Output path for metadata table.
- `age_lower` : Lower bound of patient age.
- `start_diff` : Time difference between the start of information collection period and ICU admission time(positive/negative) in hour
- `end_diff` : Time difference between the end of information collection period and ICU admission time(positive only) in hour
- For the full list of arguments, please refer to the script `1_2 Data Processing Pipeline.py`

### 2. Benchmark

<div class="figure" style="text-align: center">

<img src="figures/figure1B.png" width="85%"/>

</div>

#### 2.1 Evaluating Foundation Models as Unimodal Encoders

<div class="figure" style="text-align: center">

<img src="figures/table1.png" width="85%"/>

</div>