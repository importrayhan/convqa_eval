# ConvQA-Eval: Conversational Query Intent Detection Evaluation Suite
<p align="center">
  <img src="examples/banner.png" alt="MyProject Banner" width="60%"/>
</p>
A comprehensive evaluation framework for conversational query intent detection and ambiguity resolution, inspired by the [CoIR](https://github.com/CoIR-team/coir) architecture.

<!-- Visitor Badge -->
<p align="center">
  <img src="https://komarev.com/ghpvc/?username=importrayhan&repo=convqa_eval&style=for-the-badge&color=0f3460&label=VISITORS" alt="Visitor Count"/>
</p>

## 🎯 Overview

ConvQA-Eval provides a standardized benchmarking suite for evaluating models on conversational query understanding tasks, including:

- **Ambiguity Detection**: Identifying when user queries need clarification
- **Candidate Enumeration**: Predicting the number of valid interpretations
- **Condition Extraction**: Identifying constraints and clarification needs
- **Explanation Generation**: Producing human-readable reasoning

## 📊 Features

- **Modular Architecture**: Similar to CoIR's design with pluggable components
- **Multiple Benchmarks**: QuAC, CoQA, QReCC, and custom dataset support
- **Baseline Models**: PyTerrier RAG integration out-of-the-box
- **Comprehensive Metrics**: Accuracy, F1, MAE, BLEU
- **Easy Integration**: Simple API for adding new models and datasets

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With PyTerrier support
pip install -e ".[pyterrier]"

# Development mode
pip install -e ".[dev]"
