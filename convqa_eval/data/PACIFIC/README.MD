# PACIFIC dataset (ProActive ConversatIonal question answering in FInanCe) <br> 

📊 Dataset Overview <br> 
| Dataset | Domain  | Turn  | Modality     | Proact. | NR (Numerical Reasoning) |
| ------- | ------- | ----- | ------------ | ------- | ------------------------ |
| PACIFIC | Finance | Multi | Table / Text | Yes     | Yes                      |
### Train.json, Dev.json are annotated for "req_clari": true/false and disambiguated "original_question": "_______?"  <br> 
Open access to dataset for future studies via [github: dengyang17/PACIFIC/](https://github.com/dengyang17/PACIFIC/)<br> 

## Dataset Annotation Details:
Rigorous quality control process for high annotation quality- maintaining the complexity and difficulty of expert-level financial question answering.
## Annotators

### Qualifications
- **Well-educated postgraduate students** majored in finance or similar disciplines
- Two annotators per conversation sample with distinct roles:
  - **First annotator**: Acts as the seeker, performs primary annotation tasks
  - **Second annotator**: Acts as the agent, provides clarifying questions

## Quality Assurance Process

### Multi-Layer Verification

#### 1. **Two-Verifier System**
- Two separate verifiers validate each turn in the constructed conversations
- Verifiers check for:
  - Coherence of the conversation
  - Correctness of annotations
  - Any mistakes or problems

#### 2. **Iterative Correction Process**
- If mistakes are found, annotators must **fix issues immediately**
- Corrections continue **until the annotation passes checks by both verifiers**
- No annotation is finalized without dual verification approval

### Quality Metrics

| Metric | Value |
|--------|-------|
| **First-round mistakes captured** | 212 out of 19,008 turns (1.1%) |
| **Inter-annotator agreement** | 0.62 |

### Template-Based Objectivity

- **Pre-defined templates** provided for clarification questions
- Ensures objectivity in clarification question annotations
- Example template: *"What kind of _ are you asking about?"*
- Templates use the omitted entity to construct the clarification


## Dataset Statistics (PACIFIC / TAT-QA)

| Metric                         | Train            | Dev              | 
|--------------------------------|------------------|------------------|
| # Dialogues                    | 2,201 / -        | 278 / -          |
| # Turns (QA pairs)             | 15,087 / 13,215  | 1,982 / 1,668    | 
| # Clarifying turns             | 1,872 / -        | 320 / -          | 
| Avg. turns / dialogue          | 6.9 / -          | 7.1 / -          |
| Avg. words / question          | 9.6 / 12.5       | 9.0 / 12.4       |
| Avg. words / answer            | 4.6 / 4.1        | 4.6 / 4.1        |


Here is a sample conversation session from dev set: <br> 

```
[{
    "table": {
      "uid": "6ee29b25-8ed0-415b-858f-4147a00915f4",
      "table": [
        [
          "Product Category ",
          "Fiscal Year 2019",
          "Fiscal Year 2018 ",
          "Fiscal Year 2017"
        ],
        [
          "Fresh, vacuum-sealed chicken",
          "$1,310.2",
          "$1,139.3",
          "$1,339.1"
        ],
        [
          "Fresh, chill-packed chicken",
          "1,137.7",
          "1,158.3",
          "1,044.7"
        ],
        [
          "Fresh, ice-packed chicken ",
          "511.5",
          "503.6",
          "547.1"
        ],
        [
          "Prepared chicken ",
          "240.8",
          "207.6",
          "170.8"
        ],
        [
          "Frozen chicken ",
          "213.0",
          "211.5",
          "223.9"
        ],
        [
          "Other ",
          "27.1",
          "15.7",
          "16.6"
        ],
        [
          "Total net sales ",
          "$3,440.3",
          "$3,236.0",
          "$3,342.2"
        ]
      ]
    },
    "paragraphs": [
      {
        "uid": "ce05a041-268c-491e-8d94-2a1bc07525e9",
        "order": 1,
        "text": "2. Disaggregation of Revenue"
      },
      {
        "uid": "5b143554-0ca6-47ec-b5bc-cfec7a6221cb",
        "order": 2,
        "text": "The following table disaggregates our net sales by product category (in millions):"
      }
    ],
    "questions": [
      {
        "uid": "6463249e-391c-414c-8ed5-b082ee8148c6",
        "derivation": "",
        "answer_type": "question",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "",
        "answer_from": "table-text",
        "question": "What is the net sales from Fresh chicken for fiscal years 2019 to 2017 respectively?",
        "answer": [
          "What kind of fresh chicken are you asking about?"
        ],
        "order": 1,
        "req_clari": true,
        "follow_up": false,
        "original_question": "What is the net sales from Fresh chicken for fiscal years 2019 to 2017 respectively?"
      },
      {
        "uid": "0e6b073c-0409-48f3-aff8-66d25d01cd99",
        "order": 2,
        "question": "Fresh, vacuum-sealed chicken.",
        "answer": [
          "$1,310.2",
          "$1,139.3",
          "$1,339.1"
        ],
        "derivation": "",
        "answer_type": "multi-span",
        "answer_from": "table-text",
        "rel_paragraphs": [
          "2"
        ],
        "req_comparison": false,
        "scale": "million",
        "follow_up": false,
        "req_clari": false,
        "original_question": "What is the net sales from Fresh, vacuum-sealed chicken for fiscal years 2019 to 2017 respectively?"
      },
      {
        "uid": "b0e305cc-9d2c-4dcc-a320-bb3c3f7e6690",
        "order": 3,
        "question": "How about Fresh, chill-packed chicken?",
        "answer": [
          "1,137.7",
          "1,158.3",
          "1,044.7"
        ],
        "derivation": "",
        "answer_type": "multi-span",
        "answer_from": "table-text",
        "rel_paragraphs": [
          "2"
        ],
        "req_comparison": false,
        "scale": "million",
        "follow_up": true,
        "req_clari": false,
        "original_question": "What is the net sales from Fresh, chill-packed chicken for fiscal years 2019 to 2017 respectively?"
      },
      {
        "uid": "02213cb2-4e04-4c01-8287-81a69a9393cd",
        "order": 4,
        "question": "What is the average amount of them?",
        "answer": 1113.57,
        "derivation": "(1,137.7+1,158.3+1,044.7)/3",
        "answer_type": "arithmetic",
        "answer_from": "table-text",
        "rel_paragraphs": [
          "2"
        ],
        "req_comparison": false,
        "scale": "million",
        "follow_up": true,
        "req_clari": false,
        "original_question": "What is the average net sales from Fresh, chill-packed chicken for fiscal years 2019 to 2017?"
      },
      {
        "uid": "1ef9a3f4-53a3-48ff-ae58-cff73397769f",
        "derivation": "",
        "answer_type": "question",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "",
        "answer_from": "table-text",
        "question": "What is the net sales from Fresh, ice-packed chicken?",
        "answer": [
          "Which year are you asking about?"
        ],
        "order": 5,
        "req_clari": true,
        "follow_up": false,
        "original_question": "What is the net sales from Fresh, ice-packed chicken?"
      },
      {
        "uid": "50eb4cb7-8079-467b-ae33-a9af99e84546",
        "order": 6,
        "question": "For fiscal years 2019 to 2017 respectively.",
        "answer": [
          "511.5",
          "503.6",
          "547.1"
        ],
        "derivation": "",
        "answer_type": "multi-span",
        "answer_from": "table-text",
        "rel_paragraphs": [
          "2"
        ],
        "req_comparison": false,
        "scale": "million",
        "follow_up": false,
        "req_clari": false,
        "original_question": "What is the net sales from Fresh, ice-packed chicken for fiscal years 2019 to 2017 respectively?"
      },
      {
        "uid": "9225b117-650b-4734-ba5d-15d17279ba1f",
        "order": 7,
        "question": "What is the average of its amount for fiscal years 2019 to 2017??",
        "answer": 520.73,
        "derivation": "(511.5+503.6+547.1)/3",
        "answer_type": "arithmetic",
        "answer_from": "table-text",
        "rel_paragraphs": [
          "2"
        ],
        "req_comparison": false,
        "scale": "million",
        "follow_up": true,
        "req_clari": false,
        "original_question": "What is the average net sales from Fresh, ice-packed chicken for fiscal years 2019 to 2017?"
      },
      {
        "uid": "0eefe9da-a997-4215-a8d4-a08240049d60",
        "derivation": "",
        "answer_type": "question",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "",
        "answer_from": "table-text",
        "question": "What is the change in value of frozen chicken?",
        "answer": [
          "Which period are you asking about?"
        ],
        "order": 8,
        "req_clari": true,
        "follow_up": false,
        "original_question": "What is the change in value of frozen chicken?"
      },
      {
        "uid": "c1fa3fc3-62f6-42c9-93fc-015b9e79d1d0",
        "order": 9,
        "question": "Between fiscal year 2019 and 2018.",
        "answer": 1.5,
        "derivation": "213.0-211.5",
        "answer_type": "arithmetic",
        "answer_from": "table-text",
        "rel_paragraphs": [
          "2"
        ],
        "req_comparison": false,
        "scale": "million",
        "follow_up": false,
        "req_clari": false,
        "original_question": "What is the change in value of frozen chicken between fiscal year 2019 and 2018?"
      }
    ]
  }]
``` 


 
📈 Method Performance <br> 
| Method                  | Dev P | Dev R | Dev F1 | Test P | Test R | Test F1 |
| ----------------------- | ----- | ----- | ------ | ------ | ------ | ------- |
| BERT<sub>large</sub>    | 84.5  | 85.4  | 84.9   | 80.0   | 83.9   | 81.7    |
| RoBERTa<sub>large</sub> | 93.1  | 89.4  | 91.2   | 90.0   | 90.8   | 90.2    |
| UniPCQA (T5)            | 93.7  | 91.0  | 92.3   | 90.6   | 91.6   | 91.1    |


Possible to do- <br> 
**Multi-class ambiguity detection (2-4 classes) with detailed metadata output.** <br> 

questions[].follow_up=True → ambiguous_type=1 <br> 
questions[].req_clari=True → ambiguous_type=2 <br> 
questions[].answer_type='question' → ambiguous_type=3 <br> 
else: ambiguous_type=0 (clear intent) <br> 


## 📚 Citation <br> 

```
bibtex
@inproceedings{emnlp22-pacific,
  author    = {Yang Deng and
               Wenqiang Lei and
               Wenxuan Zhang and
               Wai Lam and
               Tat{-}Seng Chua},
  title     = {{PACIFIC:} Towards Proactive Conversational Question Answering over
               Tabular and Textual Data in Finance},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2022},
  year      = {2022},
}
```
