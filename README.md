# Image Defect Detection (MVTec AD)

This project aims to develop a quality inspection system based on Deep Learning, utilizing the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) dataset. The main goal is to automatically identify defects across various product categories, thereby contributing to improved quality control processes in the industry.

The proposed approach follows a pipeline that includes:
- **Data Preparation and Initial Analysis**  
- **Preprocessing (Resizing, Normalization, Data Augmentation)**  
- **Dataset Splitting (Training/Validation/Test)**  
- **Model Selection and Configuration (Transfer Learning, Autoencoders, etc.)**  
- **Training and Evaluation (Metrics, Monitoring, Optimization)**  
- **Production Integration (Optimization, Deployment, Real-Time Monitoring)**  

---

## Commit Standards

To keep the commit history organized and easy to understand, we have adopted the following standard, focusing on the **type** and a **brief summary**:

### 1. Type

Use one of the following types to categorize your commit:

- **feat**: new feature  
- **fix**: bug fix  
- **docs**: documentation changes  
- **style**: formatting/styling changes (without affecting functionality)  
- **refactor**: code refactoring (without changing behavior)  
- **perf**: performance improvements  
- **test**: adding or correcting tests  
- **build**: changes to the build system or dependencies  
- **chore**: maintenance tasks that do not fit into the above categories  

### 2. Brief Summary

After specifying the type, add a colon (`:`) followed by a concise description (ideally up to 72 characters) summarizing the change. For example:

```
feat: adds field validation to the form
```
