# Kubeflow Project

This project demonstrates how to use Metaflow to create and manage Kubeflow pipelines.

## Setup

This is a Poetry project. To install dependencies:

```bash
poetry install
```

## Commands

- **Run the Metaflow flow:**
```bash
poetry run python wine_pipeline.py run
```

- **Deploy to Kubeflow:**
```bash
poetry run python wine_pipeline.py kubeflow run
```