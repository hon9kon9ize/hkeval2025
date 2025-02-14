MMLU_TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
MMLU_CHOICES = ["A", "B", "C", "D"]
MMLU_SYSTEM_PROMPT = "Follow the given examples and answer the question. you should only return the answer: A, B, C, or D."
PHONETICS_TASKS = ["character_metaknowledge", "phonology"]
PHONETICS_CHOICES = ["A", "B", "C", "D", "E"]
PHONETICS_PROMPT = "Follow the given examples and answer the question. you should only return the answer: A, B, C, D, or E."

HK_LAW_TASKS = [
    "children_adoption",
    "children_protection",
    "consumer_rights",
    "discrimination",
    "domestic_violence",
    "domestic_worker",
    "juveniles_crime",
    "landlord_tenant",
    "neighborhood_disputes",
    "pet",
    "privacy",
    "property_maintenance",
    "property_redevelopment",
    "stalker",
    "youth_employment",
    "basic_law",
]
HK_LAW_SYSTEM_PROMPT = "Follow the given examples and answer the question. the question is about Hong Kong law. you should only return the answer: A, B, C, or D."

PROFESSIONAL_TASKS = [
    "boat_en",
    "boat_zh",
    "estate_agent_en",
    "estate_agent_zh",
    "estate_salespersons_en",
    "estate_salespersons_zh",
    "forex_en",
    "forex_zh",
    "hksi_1_fundamentals_of_securities_and_futures_en",
    "hksi_1_fundamentals_of_securities_and_futures_zh",
    "hksi_2_regulation_of_securities_en",
    "hksi_2_regulation_of_securities_zh",
    "hksi_3_regulation_of_derivatives_en",
    "hksi_3_regulation_of_derivatives_zh",
    "hksi_5_regulation_of_corporate_finance_en",
    "hksi_5_regulation_of_corporate_finance_zh",
    "hksi_6_regulation_of_asset_management_en",
    "hksi_6_regulation_of_asset_management_zh",
    "hksi_7_financial_markets_en",
    "hksi_7_financial_markets_zh",
    "hksi_8_securities_en",
    "hksi_8_securities_zh",
    "hksi_9_derivatives_en",
    "hksi_9_derivatives_zh",
    "hksi_11_corporate_finance_en",
    "hksi_11_corporate_finance_zh",
    "hksi_12_asset_management_en",
    "hksi_12_asset_management_zh",
    "insurance_en",
    "insurance_zh",
    "mpf_en",
    "mpf_zh",
    "taxi",
]
PROFESSIONAL_SYSTEM_PROMPT = "Follow the given examples and answer the question. the question is about professional knowledge in Hong Kong. you should only return the answer: A, B, C, or D."

DSE_TASKS = [
    "bafs_en",
    "bafs_zh",
    "bio_en",
    "bio_zh",
    "chem_en",
    "chem_zh",
    "econ_en",
    "econ_zh",
    "geog_en",
    "geog_zh",
    "ict_en",
    "ict_zh",
    "math_en",
    "math_zh",
    "phy_en",
    "phy_zh",
    "ths_zh",
]
DSE_SYSTEM_PROMPT = "Follow the given examples and answer the question. the question is about Hong Kong DSE. you should only return the answer: A, B, C, or D."

CULTURAL_TASKS = [
    "life_in_hk",
    "food",
    "history_and_landmarks",
    "langauge_and_expressions",
    "local_knowledge",
]
CULTURAL_SYSTEM_PROMPT = "Follow the given examples and answer the question. the question is about Hong Kong. Only return the answer: A, B, C, or D. DO NOT EXPLAIN."


SUMMARIZATION_PROMPT = """我哋會提供一段用廣東話寫成嘅文本。請你將文本概括成200字嘅廣東話，保留核心訊息同主題，確保內容準確、行文流暢連貫，並且忠於原意。

## 文本（原文）：
{}

## 預期輸出：
"""


ENG_YUE_ZERO_SHOT_RANSLATION_PROMPT = """將以下繁體中文句子翻譯成香港的廣東話：

{}

確保翻譯準確自然，並符合香港的廣東話語法及表達方式

只需回覆翻譯部份，不需要解釋"""


ENG_YUE_FEW_SHOTS_RANSLATION_PROMPT = """參考以下例子，將以下英文句字翻譯成香港的廣東話：

例子1：英文: {src_example1} 廣東話: {tgt_example1}
例子2：英文: {src_example2} 廣東話: {tgt_example2}
例子3：英文: {src_example3} 廣東話: {tgt_example3}

請翻譯以下文本：
{{}}

確保翻譯準確自然，並符合香港的廣東話語法及表達方式

只需回覆翻譯部份，不需要解釋"""


ZH_YUE_ZERO_SHOT_RANSLATION_PROMPT = """將以下繁體中文句子翻譯成香港的廣東話：

{}

確保翻譯準確自然，並符合香港的廣東話語法及表達方式

只需回覆翻譯部份，不需要解釋"""


ZH_YUE_FEW_SHOTS_RANSLATION_PROMPT = """參考以下例子，將以下繁體中文句子翻譯成香港的廣東話：

例子1：中文書面語: {src_example1} 廣東話: {tgt_example1}
例子2：中文書面語: {src_example2} 廣東話: {tgt_example2}
例子3：中文書面語: {src_example3} 廣東話: {tgt_example3}

請翻譯以下文本：
{{}}

確保翻譯準確自然，並符合香港的廣東話語法及表達方式

只需回覆翻譯部份，不需要解釋"""


YUE_ENG_ZERO_SHOT_RANSLATION_PROMPT = """Translate the following Cantonese text into fluent English:

{}

Ensure the translation is accurate and natural, adhering to English grammar and expression.
ONLY RETURN THE TRANSLATION. DO NOT EXPLAIN."""


YUE_ENG_FEW_SHOTS_RANSLATION_PROMPT = """Translate the following Cantonese text into English, referring to the examples below:

Example 1: Cantonese: {src_example1} English: {tgt_example1}
Example 2: Cantonese: {src_example2} English: {tgt_example2}
Example 3: Cantonese: {src_example3} English: {tgt_example3}

Text to Translate:
{{}}

Ensure the translation is accurate and natural, preserving the original meaning and using concise and fluent English expression.
ONLY RETURN THE TRANSLATION. DO NOT EXPLAIN."""


YUE_ZH_ZERO_SHOT_RANSLATION_PROMPT = """將以下廣東話句字翻譯成流暢的繁體中文書面語：

{}

確保翻譯準確自然，保留原意，並以簡潔流暢繁體中文書面語表達

只需回覆翻譯部份，不需要解釋"""


YUE_ZH_FEW_SHOTS_RANSLATION_PROMPT = """參考以下例子，將以下廣東話句字翻譯成繁體中文書面語：

例子1：廣東話: {src_example1} 繁體中文書面語: {tgt_example1}
例子2：廣東話: {src_example2} 繁體中文書面語: {tgt_example2}
例子3：廣東話: {src_example3} 繁體中文書面語: {tgt_example3}

請翻譯以下文本：
{{}}

確保翻譯準確自然，保留原意，並以簡潔流暢繁體中文書面語表達

只需回覆翻譯部份，不需要解釋"""


SENTIMENT_TASKS = ["openrice", "facebook"]

SENTIMENT_SYSTEM_PROMPT = """Follow the given examples and analyze the sentiment of the Cantonese text. You should only return the answer: Positive, Negative or Neutral.

Input: {sentiment_example1}

Sentiment: {sentiment_target1}

Input: {{}}

Sentiment:"""


benchmark_tasks = {
    "mmlu": (MMLU_TASKS, MMLU_CHOICES, MMLU_SYSTEM_PROMPT),
    "canto-mmlu": (MMLU_TASKS, MMLU_CHOICES, MMLU_SYSTEM_PROMPT),
    "phonetics": (PHONETICS_TASKS, PHONETICS_CHOICES, PHONETICS_PROMPT),
    "hk-law": (HK_LAW_TASKS, MMLU_CHOICES, HK_LAW_SYSTEM_PROMPT),
    "professional": (PROFESSIONAL_TASKS, MMLU_CHOICES, PROFESSIONAL_SYSTEM_PROMPT),
    "dse": (DSE_TASKS, MMLU_CHOICES, DSE_SYSTEM_PROMPT),
    "cultural": (CULTURAL_TASKS, MMLU_CHOICES, CULTURAL_SYSTEM_PROMPT),
}

nlp_tasks = {
    "summarization": SUMMARIZATION_PROMPT,
    "fewshot_eng_yue_translation": ENG_YUE_FEW_SHOTS_RANSLATION_PROMPT,
    "fewshot_yue_eng_translation": YUE_ENG_FEW_SHOTS_RANSLATION_PROMPT,
    "fewshot_yue_zh_translation": YUE_ZH_FEW_SHOTS_RANSLATION_PROMPT,
    "fewshot_zh_yue_translation": ZH_YUE_FEW_SHOTS_RANSLATION_PROMPT,
    "eng_yue_translation": ENG_YUE_ZERO_SHOT_RANSLATION_PROMPT,
    "yue_eng_translation": YUE_ENG_ZERO_SHOT_RANSLATION_PROMPT,
    "yue_zh_translation": YUE_ZH_ZERO_SHOT_RANSLATION_PROMPT,
    "zh_yue_translation": ZH_YUE_ZERO_SHOT_RANSLATION_PROMPT,
    "sentiment": SENTIMENT_SYSTEM_PROMPT,
}


ZH_JYUTPING_PROMPT = """You are an expert in Cantonese linguistics. Please convert the given Cantonese sentence into Jyutping romanisation. You can ignore all punctuation marks, and normalize all numerals and English loanwords into Cantonese pronunciation. Do not include any further explanation.
Example 1
{src_example1}
{tgt_example1}

Example 2
{src_example2}
{tgt_example2}

Example 3
{src_example3}
{tgt_example3}

Example 4
{src_example4}
{tgt_example4}

Example 5
{src_example5}
{tgt_example5}

Text to Convert: 
{{}}
"""


phonetics_g2p_tasks = {
    "zh_to_jyutping": ZH_JYUTPING_PROMPT,
}
