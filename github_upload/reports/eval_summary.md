# Evaluation summary

Prompt version: 2.0.

Primary metric per task (0-1, higher is better): calculation = numeric answer correct; structured_json = valid JSON with the correct risk band; refusal_* = correctly declined; concept_qa = ROUGE-L vs reference; all others = fraction of required facts present in the answer.

| task_type              |   base |   base+prompt |   base+rag |   base+rag+prompt |   tuned |   tuned+rag |
|:-----------------------|-------:|--------------:|-----------:|------------------:|--------:|------------:|
| calculation            |  0.167 |         0.208 |    nan     |           nan     |   0.375 |      nan    |
| concept_qa             |  0.175 |         0.226 |      0.346 |             0.331 |   0.842 |        1    |
| group_fact             |  0     |         0     |      0.75  |             0.75  |   0     |        1    |
| product_fact           |  0.04  |         0     |      0.48  |             0.76  |   0.04  |        0.92 |
| product_recommendation |  0.75  |         0.65  |    nan     |           nan     |   1     |      nan    |
| refusal_no_context     |  0.5   |         0.1   |    nan     |           nan     |   1     |      nan    |
| refusal_wrong_context  |  0     |         0     |    nan     |           nan     |   1     |      nan    |
| risk_interpretation    |  0.471 |         0.971 |    nan     |           nan     |   0.971 |      nan    |
| scenario_analysis      |  0     |         0.5   |    nan     |           nan     |   1     |      nan    |
| structured_json        |  0.529 |         0.765 |    nan     |           nan     |   1     |      nan    |

## Average across tasks (unweighted mean of examples)

| config          |   mean_primary_metric |
|:----------------|----------------------:|
| base            |                 0.263 |
| base+prompt     |                 0.342 |
| base+rag        |                 0.438 |
| base+rag+prompt |                 0.561 |
| tuned           |                 0.679 |
| tuned+rag       |                 0.963 |

## Effect of the engineered prompt (difference in primary metric; positive = engineered prompt is better)

| task_type              |   prompt engineering alone (no fine-tuning) |   prompt engineering on top of RAG |
|:-----------------------|--------------------------------------------:|-----------------------------------:|
| calculation            |                                       0.041 |                            nan     |
| concept_qa             |                                       0.051 |                             -0.015 |
| group_fact             |                                       0     |                              0     |
| product_fact           |                                      -0.04  |                              0.28  |
| product_recommendation |                                      -0.1   |                            nan     |
| refusal_no_context     |                                      -0.4   |                            nan     |
| refusal_wrong_context  |                                       0     |                            nan     |
| risk_interpretation    |                                       0.5   |                            nan     |
| scenario_analysis      |                                       0.5   |                            nan     |
| structured_json        |                                       0.236 |                            nan     |

## Latency and tokens (per generation)

| config          |   latency_mean_s |   latency_p50_s |   latency_p95_s |   new_tokens_mean |   prompt_tokens_mean |
|:----------------|-----------------:|----------------:|----------------:|------------------:|---------------------:|
| base            |             7.13 |            7.58 |           10.31 |            140.17 |               129.47 |
| base+prompt     |             3.76 |            3    |            9.03 |             64.56 |               527.45 |
| base+rag        |             6.08 |            4.88 |           10.63 |            111.93 |               420.26 |
| base+rag+prompt |             3.26 |            2.81 |            7.19 |             47.94 |               851.09 |
| tuned           |             4.03 |            3.6  |            8.81 |             56.52 |               129.47 |
| tuned+rag       |             3.18 |            3.21 |            4.32 |             39.44 |               420.26 |

## Share of product/group-fact questions where the model declined to answer

| config          |   refused |
|:----------------|----------:|
| base            |     0.31  |
| base+prompt     |     0.172 |
| base+rag        |     0.172 |
| base+rag+prompt |     0.034 |
| tuned           |     0.793 |
| tuned+rag       |     0     |
