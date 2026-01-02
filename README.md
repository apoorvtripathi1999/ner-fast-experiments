# Biomedical NER with LLMs: Zero-Shot Experiments

## Research Perspective

This project explores the application of Large Language Models (LLMs) for Biomedical Named Entity Recognition (NER), specifically focusing on disease identification in biomedical texts. The research investigates zero-shot learning capabilities of various LLMs to perform NER tasks without traditional fine-tuning, leveraging the inherent knowledge and reasoning abilities of these models.

The primary goal is to evaluate how different LLMs perform on biomedical NER tasks, particularly in identifying disease entities, and to establish a baseline for future improvements through advanced architectures like multi-agent systems with Retrieval-Augmented Generation (RAG).

## Current Approach

The project currently implements zero-shot classification using five different LLMs:

- **GPT-120B**: A large-scale GPT model
- **GPT-20B**: A smaller GPT variant
- **Llama-70B**: Meta's large Llama model
- **Llama-8B**: Meta's smaller Llama model
- **Quen-32B**: A 32B parameter model

Each notebook performs zero-shot NER by processing text in chunks of 10-20 tokens per LLM call, classifying each token as either "Disease" or "O" (outside entity). The approach uses prompt engineering to guide the models in identifying disease mentions without explicit training data.

### Dataset
The experiments use a biomedical NER dataset with token-level annotations for disease entities. The dataset contains approximately 135,971 tokens, with a small proportion being disease entities (around 8% in the sampled data).

## Findings

All experiments were conducted on a sample of 500 tokens containing 40 disease entities and 460 non-entity tokens.

### Model Performance Comparison

| Model      | Accuracy | F1 Score (Disease) | Precision (Disease) | Recall (Disease) |
|------------|----------|-------------------|---------------------|------------------|
| Llama-70B | 0.97    | 0.806            | 1.00               | 0.68            |
| GPT-120B  | 0.95    | 0.596            | 1.00               | 0.42            |
| GPT-20B   | 0.95    | 0.596            | 1.00               | 0.42            |
| Quen-32B  | 0.95    | 0.519            | 1.00               | 0.35            |
| Llama-8B  | 0.89    | 0.454            | -                  | -               |

### Key Observations

1. **High Precision, Low Recall**: All models show perfect precision (1.00) for disease entities, meaning when they predict a disease, it's almost always correct. However, recall varies significantly, indicating models miss many actual disease entities.

2. **Model Size Impact**: Larger models generally perform better. Llama-70B achieves the highest F1 score (0.806) and accuracy (0.97), while Llama-8B performs worst (F1: 0.454, Accuracy: 0.89).

3. **GPT Models**: Both GPT-120B and GPT-20B show identical performance (F1: 0.596, Accuracy: 0.95), suggesting that for this task, model size beyond 20B parameters doesn't provide additional benefits in zero-shot setting.

4. **Overall Accuracy**: Despite low recall for disease entities, overall accuracy remains high (89-97%) due to the class imbalance - most tokens are non-entities.

5. **Under-prediction**: All models tend to under-predict disease entities, with predicted disease counts ranging from 14 (Quen-32B) to 27 (Llama-70B) compared to 40 true positives.

## Future Work

The next phase of this research will implement a **Multi-RAG Multi-Agent System** with a coordinator to address current limitations:

### Planned Improvements

1. **Multi-Agent Architecture**: 
   - Coordinator agent to orchestrate the NER process
   - Specialized agents for different biomedical domains (gene mutations, diseases, drugs)
   - Agent communication and consensus mechanisms

2. **Retrieval-Augmented Generation (RAG)**:
   - Integration of external biomedical knowledge bases
   - Context-aware entity recognition using domain-specific literature
   - Dynamic prompt adaptation based on retrieved information

3. **Domain-Specific NER**:
   - Gene mutation identification
   - Disease entity recognition
   - Drug name extraction
   - Chemical compound detection

4. **Enhanced Metrics**:
   - Improved recall while maintaining precision
   - Domain-specific performance evaluation
   - Cross-domain generalization testing

### Expected Benefits

- **Higher Recall**: RAG components should help identify more disease entities by providing contextual biomedical knowledge
- **Domain Adaptability**: Multi-agent system can specialize in different biomedical sub-domains
- **Robustness**: Coordinator can resolve conflicts and improve overall accuracy
- **Scalability**: Modular architecture allows easy addition of new domains and models

## Project Structure

```
ner-fast-experiments/
├── gpt-120b.ipynb      # GPT-120B zero-shot experiments
├── gpt-20b.ipynb       # GPT-20B zero-shot experiments
├── llama-70b.ipynb     # Llama-70B zero-shot experiments
├── llama-8b.ipynb      # Llama-8B zero-shot experiments
└── quen-32b.ipynb      # Quen-32B zero-shot experiments
```

## Requirements

- Python 3.8+
- Required packages: pandas, scikit-learn, transformers, torch
- Access to respective LLM APIs or local deployments

## Usage

Each notebook can be run independently. Ensure you have the necessary API keys or model access configured before execution.
