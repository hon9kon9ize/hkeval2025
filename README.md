# HKCanto-Eval: A Benchmark for Evaluating Cantonese Language Understanding and Cultural Comprehension in LLMs

## Usage

### Run Evaluation

```
python scripts/eval.py \
    --model hon9kon9ize/CantoneseLLMChat-v0.5
    --data_dir data/benchmark_data/MMLU
    --ntrain 5 # optional, default 5
```


## Benchmark

### Cultural

| categories | CantonesLLMChat-v0.5-6b | Claude Sonnet 3.5 | GPT4o-Mini | Gemini 1.5 Flash |
| --- | --- | --- | --- | --- |
| average | 0.5804 | 0.7768 | 0.6562 | 0.6830 |
| life_in_hk | 0.5571 | 0.8286 | 0.7000 | 0.7714 |
| food | 0.5741 | 0.7037 | 0.6481 | 0.6296 |
| history_and_landmarks | 0.6071 | 0.8214 | 0.6786 | 0.6964 |
| langauge_and_expressions | 0.5909 | 0.7273 | 0.5682 | 0.5909 |

### DSE

| categories | CantonesLLMChat-v0.5-6b | Claude Sonnet 3.5 | GPT4o-Mini | Gemini 1.5 Flash |
| --- | --- | --- | --- | --- |
| average | 0.4417 | 0.4605 | 0.3977 | 0.4090 |
| bafs_en | 0.4222 | 0.8222 | 0.5778 | 0.7556 |
| bafs_zh | 0.4000 | 0.7333 | 0.5111 | 0.6889 |
| bio_en | 0.6383 | 0.7660 | 0.7021 | 0.7660 |
| bio_zh | 0.4894 | 0.8298 | 0.7660 | 0.7234 |
| chem_en | 0.3784 | 0.7297 | 0.5946 | 0.7838 |
| chem_zh | 0.2973 | 0.6486 | 0.4595 | 0.6486 |
| econ_en | 0.3902 | 0.7317 | 0.5122 | 0.4390 |
| econ_zh | 0.3659 | 0.6585 | 0.4390 | 0.4878 |
| geog_en | 0.4694 | 0.0000 | 0.0204 | 0.0000 |
| geog_zh | 0.4490 | 0.0000 | 0.0408 | 0.0000 |
| ict_en | 0.6471 | 0.0000 | 0.1324 | 0.0000 |
| ict_zh | 0.5441 | 0.0000 | 0.0294 | 0.0000 |
| math_en | 0.3103 | 0.3103 | 0.3621 | 0.2586 |
| math_zh | 0.2414 | 0.3103 | 0.3448 | 0.2586 |
| phy_en | 0.3200 | 0.7200 | 0.5600 | 0.5600 |
| phy_zh | 0.3200 | 0.5600 | 0.6000 | 0.5600 |
| ths_zh | 0.5614 | 0.8070 | 0.6491 | 0.7368 |

### HK Law

| categories | CantonesLLMChat-v0.5-6b | GPT4o-Mini | Gemini 1.5 Flash | Claude Sonnet 3.5 |
| --- | --- | --- | --- | --- |
| average | 0.7028 | 0.8083 | 0.7694 | 0.8514 |
| children_adoption | 0.5500 | 0.7000 | 0.7500 | 0.7000 |
| children_protection | 1.0000 | 0.9333 | 0.9333 | 1.0000 |
| consumer_rights | 0.5500 | 0.7500 | 0.8500 | 0.8000 |
| discrimination | 0.9000 | 0.8000 | 0.7333 | 0.8667 |
| domestic_violence | 0.8000 | 0.9200 | 0.8800 | 0.8400 |
| domestic_worker | 0.6000 | 0.7667 | 0.8000 | 0.9000 |
| juveniles_crime | 0.6000 | 0.8000 | 0.6800 | 0.7200 |
| landlord_tenant | 0.6889 | 0.8000 | 0.7111 | 0.8222 |
| neighborhood_disputes | 0.7000 | 0.8000 | 0.7500 | 0.8000 |
| pet | 0.8000 | 0.8857 | 0.8286 | 0.9143 |
| privacy | 0.7500 | 0.8500 | 0.9000 | 0.7500 |
| property_maintenance | 0.6000 | 0.8500 | 0.7000 | 1.0000 |
| property_redevelopment | 0.7500 | 0.9000 | 0.7000 | 1.0000 |
| stalker | 0.6000 | 0.6667 | 0.6000 | 0.7333 |
| youth_employment | 0.6000 | 0.6500 | 0.7500 | 0.7500 |

### Phonetic

| categories | CantonesLLMChat-v0.5-6b | GPT4o-Mini | Gemini 1.5 Flash | Claude Sonnet 3.5 |
| --- | --- | --- | --- | --- |
| average | 0.2500 | 0.2600 | 0.2800 | 0.3900 |
| character_metaknowledge | 0.3100 | 0.1700 | 0.4200 | 0.5300 |
| phonology | 0.1900 | 0.2600 | 0.1400 | 0.2500 |

### MMLU

| categories | CantonesLLMChat-v0.5-6b | GPT4o-Mini | Gemini 1.5 Flash | Gemini 1.5 Pro |
| --- | --- | --- | --- | --- |
| average | 0.5833 | 0.7582 | 0.7821 | 0.8316 |
| abstract_algebra | 0.3400 | 0.4100 | 0.5200 | 0.7200 |
| anatomy | 0.5481 | 0.8148 | 0.7852 | 0.7926 |
| astronomy | 0.6382 | 0.8618 | 0.8947 | 0.9211 |
| business_ethics | 0.6400 | 0.7800 | 0.8100 | 0.8600 |
| clinical_knowledge | 0.6226 | 0.8528 | 0.8340 | 0.8377 |
| college_biology | 0.6528 | 0.9028 | 0.9375 | 0.9028 |
| college_chemistry | 0.4300 | 0.5400 | 0.6300 | 0.6500 |
| college_computer_science | 0.5100 | 0.6500 | 0.6800 | 0.7900 |
| college_mathematics | 0.4300 | 0.4400 | 0.5300 | 0.6900 |
| college_medicine | 0.5723 | 0.7630 | 0.7572 | 0.8035 |
| college_physics | 0.3922 | 0.5686 | 0.5980 | 0.7059 |
| computer_security | 0.7400 | 0.8400 | 0.8000 | 0.7900 |
| conceptual_physics | 0.5915 | 0.7830 | 0.8553 | 0.9277 |
| econometrics | 0.3684 | 0.6667 | 0.6053 | 0.7193 |
| electrical_engineering | 0.6276 | 0.7586 | 0.7862 | 0.8000 |
| elementary_mathematics | 0.4735 | 0.6111 | 0.7989 | 0.9312 |
| formal_logic | 0.4444 | 0.5397 | 0.6429 | 0.7222 |
| global_facts | 0.3700 | 0.5100 | 0.5400 | 0.5500 |
| high_school_biology | 0.7323 | 0.9226 | 0.9387 | 0.9226 |
| high_school_chemistry | 0.4926 | 0.7143 | 0.7340 | 0.7438 |
| high_school_computer_science | 0.6700 | 0.8800 | 0.8700 | 0.9300 |
| high_school_european_history | 0.7576 | 0.8667 | 0.8364 | 0.8788 |
| high_school_geography | 0.7727 | 0.9495 | 0.9242 | 0.9444 |
| high_school_government_and_politics | 0.7927 | 0.9637 | 0.9845 | 0.9793 |
| high_school_macroeconomics | 0.6436 | 0.8333 | 0.8513 | 0.8692 |
| high_school_mathematics | 0.3185 | 0.3963 | 0.5074 | 0.7222 |
| high_school_microeconomics | 0.7689 | 0.9076 | 0.9034 | 0.9328 |
| high_school_physics | 0.4172 | 0.5762 | 0.6821 | 0.8146 |
| high_school_psychology | 0.7890 | 0.9229 | 0.9468 | 0.9339 |
| high_school_statistics | 0.5278 | 0.6806 | 0.7454 | 0.8565 |
| high_school_us_history | 0.7108 | 0.9020 | 0.9167 | 0.9167 |
| high_school_world_history | 0.7089 | 0.8987 | 0.9156 | 0.9325 |
| human_aging | 0.6413 | 0.8117 | 0.7892 | 0.7982 |
| human_sexuality | 0.6336 | 0.8779 | 0.8702 | 0.8702 |
| international_law | 0.7190 | 0.9091 | 0.8843 | 0.9091 |
| jurisprudence | 0.7037 | 0.8611 | 0.8796 | 0.8796 |
| logical_fallacies | 0.6503 | 0.8528 | 0.8650 | 0.9141 |
| machine_learning | 0.3929 | 0.6071 | 0.5714 | 0.7143 |
| management | 0.7864 | 0.8447 | 0.8447 | 0.9126 |
| marketing | 0.8333 | 0.9444 | 0.9231 | 0.9402 |
| medical_genetics | 0.6200 | 0.8900 | 0.8500 | 0.9200 |
| miscellaneous | 0.7395 | 0.9208 | 0.8838 | 0.9579 |
| moral_disputes | 0.6069 | 0.8006 | 0.8208 | 0.8237 |
| moral_scenarios | 0.3140 | 0.5084 | 0.6637 | 0.7709 |
| nutrition | 0.6275 | 0.8170 | 0.7908 | 0.8660 |
| philosophy | 0.6334 | 0.7846 | 0.8135 | 0.8778 |
| prehistory | 0.6111 | 0.8364 | 0.8735 | 0.8858 |
| professional_accounting | 0.4504 | 0.6277 | 0.6277 | 0.6560 |
| professional_law | 0.4263 | 0.5965 | 0.5991 | 0.6565 |
| professional_medicine | 0.5699 | 0.8603 | 0.7978 | 0.8860 |
| professional_psychology | 0.5931 | 0.8448 | 0.8415 | 0.8922 |
| public_relations | 0.6091 | 0.7727 | 0.7818 | 0.7818 |
| security_studies | 0.6816 | 0.7918 | 0.8327 | 0.8571 |
| sociology | 0.7861 | 0.9055 | 0.8955 | 0.9005 |
| us_foreign_policy | 0.7700 | 0.9100 | 0.9300 | 0.9300 |
| virology | 0.4398 | 0.5542 | 0.5663 | 0.5542 |
| world_religions | 0.7544 | 0.8655 | 0.8772 | 0.8655 |

### Canto-MMLU

| categories | CantonesLLMChat-v0.5-6b | GPT4o-Mini | Gemini 1.5 Flash | Gemini 1.5 Pro |
| --- | --- | --- | --- | --- |
| average | 0.5107 | 0.6775 | 0.7176 | 0.7599
| abstract_algebra | 0.3400 | 0.4200 | 0.4700 | 0.6400
| anatomy | 0.4593 | 0.7111 | 0.7259 | 0.7333
| astronomy | 0.5263 | 0.8355 | 0.8750 | 0.9013
| business_ethics | 0.6400 | 0.7600 | 0.7200 | 0.8200
| clinical_knowledge | 0.5698 | 0.8000 | 0.7585 | 0.7094
| college_biology | 0.4861 | 0.7986 | 0.8611 | 0.7847
| college_chemistry | 0.4100 | 0.4900 | 0.5300 | 0.6300
| college_computer_science | 0.5200 | 0.5700 | 0.6600 | 0.7800
| college_mathematics | 0.3600 | 0.4000 | 0.5300 | 0.6100
| college_medicine | 0.5260 | 0.7052 | 0.7514 | 0.7803
| college_physics | 0.3333 | 0.4510 | 0.5294 | 0.7647
| computer_security | 0.6100 | 0.7800 | 0.7500 | 0.7400
| conceptual_physics | 0.4936 | 0.7319 | 0.8128 | 0.8723
| econometrics | 0.3070 | 0.5175 | 0.5965 | 0.6754
| electrical_engineering | 0.5586 | 0.6897 | 0.7310 | 0.7310
| elementary_mathematics | 0.4577 | 0.5714 | 0.7434 | 0.8862
| formal_logic | 0.4127 | 0.4444 | 0.5714 | 0.6190
| global_facts | 0.3100 | 0.4400 | 0.4600 | 0.5000
| high_school_biology | 0.6194 | 0.8581 | 0.9452 | 0.9097
| high_school_chemistry | 0.4089 | 0.6502 | 0.7044 | 0.7685
| high_school_computer_science | 0.6200 | 0.8300 | 0.8400 | 0.9100
| high_school_european_history | 0.6545 | 0.8182 | 0.8242 | 0.5394
| high_school_geography | 0.6869 | 0.8535 | 0.8636 | 0.8939
| high_school_government_and_politics | 0.6269 | 0.8964 | 0.9119 | 0.9119
| high_school_macroeconomics | 0.5667 | 0.7872 | 0.8179 | 0.8154
| high_school_mathematics | 0.3037 | 0.3852 | 0.5222 | 0.7000
| high_school_microeconomics | 0.6176 | 0.8403 | 0.8529 | 0.8361
| high_school_physics | 0.3377 | 0.4967 | 0.6026 | 0.7815
| high_school_psychology | 0.6807 | 0.8899 | 0.8826 | 0.9284
| high_school_statistics | 0.4074 | 0.6157 | 0.6944 | 0.8241
| high_school_us_history | 0.6176 | 0.8578 | 0.8480 | 0.8775
| high_school_world_history | 0.6920 | 0.8523 | 0.8354 | 0.8734
| human_aging | 0.5874 | 0.7534 | 0.7399 | 0.7534
| human_sexuality | 0.5725 | 0.7863 | 0.7786 | 0.8168
| international_law | 0.6364 | 0.8182 | 0.8099 | 0.9008
| jurisprudence | 0.6759 | 0.7963 | 0.8426 | 0.7870
| logical_fallacies | 0.6319 | 0.7178 | 0.7730 | 0.7485
| machine_learning | 0.3571 | 0.5625 | 0.5625 | 0.6696
| management | 0.6699 | 0.7864 | 0.7767 | 0.8447
| marketing | 0.7265 | 0.8761 | 0.9017 | 0.8803
| medical_genetics | 0.5500 | 0.7700 | 0.7800 | 0.7000
| miscellaneous | 0.6564 | 0.8544 | 0.8365 | 0.8825
| moral_disputes | 0.4855 | 0.7225 | 0.7081 | 0.7717
| moral_scenarios | 0.2615 | 0.3408 | 0.5508 | 0.5385
| nutrition | 0.5654 | 0.8039 | 0.7582 | 0.8137
| philosophy | 0.5723 | 0.6817 | 0.7299 | 0.8167
| prehistory | 0.5309 | 0.6883 | 0.7685 | 0.8611
| professional_accounting | 0.3936 | 0.8039 | 0.5071 | 0.6560
| professional_law | 0.3677 | 0.6817 | 0.5117 | 0.5763
| professional_medicine | 0.4044 | 0.6883 | 0.7132 | 0.7647
| professional_psychology | 0.5343 | 0.4894 | 0.7386 | 0.7974
| public_relations | 0.5455 | 0.6727 | 0.7364 | 0.7273
| security_studies | 0.6204 | 0.7020 | 0.7714 | 0.7837
| sociology | 0.7413 | 0.8308 | 0.8458 | 0.8507
| us_foreign_policy | 0.6800 | 0.8700 | 0.8900 | 0.8800
| virology | 0.4096 | 0.4880 | 0.5482 | 0.5422
| world_religions | 0.6667 | 0.8538 | 0.8129 | 0.8480

## Overview

## Leaderboard

## Dataset

## Evaluation Process

## Citation

## Contributors
