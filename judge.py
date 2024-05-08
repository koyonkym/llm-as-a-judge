import re
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import settings


tqdm.pandas()
pd.set_option("display.max_colwidth", None)

repo_id = settings.REPO_ID

model = AutoModelForCausalLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

llm_client = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)

try:
    ratings = pd.read_csv(settings.RATINGS_CSV)
except AttributeError:
    ratings = pd.read_csv("ratings.csv")

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None

JUDGE_PROMPTS = []
for i in range(10):
    try:
        JUDGE_PROMPTS.append(getattr(settings, "JUDGE_PROMPT_" + str(i+1)))
    except AttributeError:
        break

for i in range(len(JUDGE_PROMPTS)):
    print("Judge", str(i+1))
    ratings["llm_judge_" + str(i+1)] = ratings.progress_apply(
        lambda x: llm_client(
            JUDGE_PROMPTS[i].format(question=x["question"], answer=x["answer"]),
            return_full_text=False,
        )[0]['generated_text'],
        axis=1,
    )
    ratings["llm_judge_score_" + str(i+1)] = ratings["llm_judge_" + str(i+1)].apply(extract_judge_score)
    # Get the non-null rows for both columns
    non_null_rows = ratings[["llm_judge_score_" + str(i+1), "score_" + str(i+1)]].dropna()
    print("Correlation between LLM-as-a-judge and the human raters:")
    if len(non_null_rows) > 1 and non_null_rows["llm_judge_score_" + str(i+1)].std() != 0 and non_null_rows["score_" + str(i+1)].std() != 0:
        print(f"{non_null_rows['llm_judge_score_' + str(i+1)].corr(non_null_rows['score_' + str(i+1)], method='pearson'):.3f}")
    else:
        print("No valid data points for correlation calculation")

ratings.to_csv('out.csv', index=False)

# errors = pd.concat(
#     [
#         examples.loc[examples["llm_judge_score"] > examples["human_score"]].head(1),
#         examples.loc[examples["llm_judge_score"] < examples["human_score"]].head(2),
#     ]
# )

# print(
#     errors[
#         [
#             "question",
#             "answer",
#             "human_score",
#             "explanation_1",
#             "llm_judge_score",
#             "llm_judge",
#         ]
#     ]
# )