Constraint Optimization Dataset Builder
Overview

This system is designed to help generate and augment datasets for constraint optimization problems. It supports the creation of datasets based on chaotic mapping techniques and enables the generation of two types of datasets: SFT (Supervised Fine-Tuning) augmented data and DPO (Decision Preference Optimization) datasets. The system integrates a large language model (LLM) for transforming problem descriptions, models, and code, and allows for the customization of various parameters related to the generation process.

Key Features

Dataset Management:

Add, edit, and delete samples in the dataset.

Import datasets via CSV files, ensuring the format includes IDs, problem descriptions, models, and code.

Export datasets in JSON format for further use.

LLM Integration:

Configure LLM settings (e.g., API provider, model name, and API keys).

Test the API connection to ensure proper integration.

Enable LLM-style rephrasing for enhancing problem descriptions, models, and code.

Chaotic Mapping Parameters:

Users can define parameters like r, x0, alphaMax, and the number of iterations for generating chaotic sequences.

Chaotic mappings are used to perturb the dataset and generate varied examples.

Dataset Augmentation:

Generate augmented SFT datasets using perturbation techniques on the description, model, and code.

Utilize the chaotic sequence to create realistic variations in the dataset.

Allow for LLM processing to refine augmented data.

DPO Dataset Generation:

Create DPO preference pairs by generating multiple candidate solutions for a problem.

Evaluate these candidates based on quality metrics like simplicity, readability, and efficiency, incorporating LLM-based scores if available.

Data Export:

Export both SFT and DPO datasets in JSON format for use in training or other applications.

User Interface:

Clean and interactive UI with language toggle support (Chinese and English).

Interactive tables to manage samples, view details, and perform operations like adding or deleting samples.

Real-time processing feedback and status updates.

Usage Workflow:

Dataset Creation: Start by adding base samples, either manually or by uploading a CSV file.

Augmentation: Once the base dataset is prepared, generate augmented data through chaotic perturbations.

DPO Generation: From the augmented dataset, generate DPO pairs for preference-based optimization problems.

Export Data: Export the final datasets (SFT or DPO) in JSON format for further use.