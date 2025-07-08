Benchmarking Foundation Models with Multimodal Public Electronic Health Records
=========================
## Table of contents
* [Introduction](#introduction)
* [Structure](#structure)
* [Data Requirements](#data-requirements)

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
MIMIC-CXR https://physionet.org/content/mimic-cxr/2.0.0/#files-panel
MIMIC-IV-Note https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel