# RTBoost: Boosting Repository-level Code Translation via Incremental Alignment and Alignment-Based Retrieval

## Overview


In this paper, we focus on the incremental copilot mode code translation and propose RTBoost, a retrieval-augmented method that enhances context consistency through incremental alignment and alignment retrieval. 
Specifically, RTBoost operates in a human-in-the-loop setting, where at each translation step, it aligns the previously human-verified target repository with the source repository.
Given the partially verified target repository and the complete source repository, RTBoost retrieves (1) necessary dependencies within the partially translated repository, guiding the LLM on what to use, and (2) the most similar aligned pair as a one-shot example, guiding the LLM on how to translate at the current step.
To evaluate the performance of RTBoost, we conduct experiments on 1,000 Java ↔ C# method pairs from 7 pairs of repositories and 300 Java ↔ ArkTS method pairs from 10 pairs of repositories. 
Experimental results based on the dataset demonstrate the effectiveness of RTBoost.

## Project Structure

The structure of this project is shown as follows:

```
├─ RepoTrans    # Repository context aware code translation dataset
    ├─ java_arkts    # Parallel dataset for the translation between Java and ArkTS
        ├─ java_arkts.xlsx    # Java ↔ ArkTS translation pairs with corresponding position in repository
        ├─ camera.zip    # The followings are repositories of the translation pairs, each of which contains a Java and an ArkTS sub-repository for translation
        ├─ file_selector.zip
        ├─ image_picker.zip
        ├─ local_auth.zip
        ├─ path_provider.zip
        ├─ pigeon.zip
        ├─ shared_preferences.zip
        ├─ url_launcher.zip
        ├─ video_player.zip
        └─ webview_flutter.zip
    ├─ java_csharp    # Parallel dataset for the translation between Java and C#
        ├─ java_csharp.xlsx    # Java ↔ C# translation pairs with corresponding position in repository
        ├─ antlr.zip    # The followings are repositories of the translation pairs, each of which contains a Java and an C# sub-repository for translation
        ├─ aws.zip
        ├─ jgit.zip
        ├─ lucene.zip
        ├─ openapi.zip
        ├─ poi.zip
        ├─ xobotos.zip
├─ prompt    # Chain-of-though examples used for CoT-based translation
    ├─ java2arkts_cot.txt
    ├─ arkts2java_cot.txt
    ├─ java2csharp_cot.txt
    ├─ csharp2java_cot.txt
├─ rdfs    # Python package for generating the repository dependency fractal structure (RDFS)
├─ graph_aligner.py    # Generate the RDFS of two repository and construct an alignment between them
├─ incremental_task_builder.py    # Incremental copilot translation simulation
├─ retriever.py    # Retrieve the relevant code snippets from the repositories and the constructed alignment 
├─ generation.py    # (Main entry) Generate the translation result for a given method from source repository
├─ bleu   # Python package for BLEU and CodeBLEU calculation from the repository of its original paper
├─ metrics.py    # Calculate the metrics used in the paper
├─ requirements.txt    # List of packages required to run RTBoost
└─ utils.py    # Other tools for extracting code from markdown block and file reading
```

## Quick Start

#### Install Requirements

```
pip install -r requirements.txt
```

#### RTBoost

There are 5 input arguments for RTBoost:

  - `model`: We can choose from Claude, DeepSeek, and GPT families.
  - `source_language`: The source language of translation, which can be chosen from `java`, `csharp`, and `arkts`
  - `target_language`: The target language of translation, which can be chosen from `java`, `csharp`, and `arkts`
  - `alpha`: The interpolation factor between lexical and structural similarity is set to 0.6 by default for all experiments.
    
An example for running RTBoost

```
python generation.py --model gpt-4o-2024-11-20 --source_language java --target_language arkts --mode graphtranslator --alpha 0.6
```
