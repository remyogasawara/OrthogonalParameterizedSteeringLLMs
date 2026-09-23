import sys, os
import argparse

# SET TO RUN
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle
import time
from src.gpt_evals import run_gpt_eval_on_experiment

with open("experiment_results/mislabel/Llama-3.2-3B-Instruct_layer13_coordinate-other-ais_generations.pkl", "rb") as f:
    mislabel_llama_coord_other_ai_gpt = pickle.load(f)

with open("experiment_results/mislabel/Llama-3.2-3B-Instruct_layer13_uncorrigible-neutral-HHH_generations.pkl", "rb") as f:
    mislabel_llama_uncorrigible_neutral_hhh_gpt = pickle.load(f)

with open("experiment_results/synthetic_corruption/Llama-3.2-3B-Instruct_layer13_coordinate-other-ais_generations.pkl"  , "rb") as f:
    synthetic_llama_coord_other_ai_gpt = pickle.load(f)

with open("experiment_results/synthetic_corruption/Llama-3.2-3B-Instruct_layer13_uncorrigible-neutral-HHH_generations.pkl", "rb") as f:
    synthetic_llama_uncorrigible_neutral_hhh_gpt = pickle.load(f)

with open("experiment_results/behavior_injection/Llama-3.2-3B-Instruct_layer13_coordinate-other-ais_uncorrigible-neutral-HHH_generations.pkl", "rb") as f:
    anticor_llama_coord_other_ai_uncorrigible_neutral_hhh_gpt = pickle.load(f)

with open("experiment_results/behavior_injection/Llama-3.2-3B-Instruct_layer13_myopic-reward_power-seeking-inclination_generations.pkl", "rb") as f:
    uncor_injection_llama_myopic_reward_power_seeking_inclination_gpt = pickle.load(f)

with open("experiment_results/behavior_injection/Llama-3.2-3B-Instruct_layer13_wealth-seeking-inclination_power-seeking-inclination_generations.pkl", "rb") as f:
    cor_injection_llama_wealth_seeking_inclination_power_seeking_inclination_gpt = pickle.load(f)

exps = {
    "mislabel_llama_coord_other_ai": mislabel_llama_coord_other_ai_gpt,
    "synthetic_llama_coord_other_ai": synthetic_llama_coord_other_ai_gpt,
    "anticor_llama_coord_other_ai_uncorrigible_neutral_hhh": anticor_llama_coord_other_ai_uncorrigible_neutral_hhh_gpt,
    "uncor_injection_llama_myopic_reward_power_seeking_inclination": uncor_injection_llama_myopic_reward_power_seeking_inclination_gpt,
    "cor_injection_llama_wealth_seeking_inclination_power_seeking_inclination": cor_injection_llama_wealth_seeking_inclination_power_seeking_inclination_gpt,
    "mislabel_llama_uncorrigible_neutral_hhh": mislabel_llama_uncorrigible_neutral_hhh_gpt,
    "synthetic_llama_uncorrigible_neutral_hhh": synthetic_llama_uncorrigible_neutral_hhh_gpt,
}

master_start_time = time.time()
for name, exp in exps.items():
    start_time = time.time()
    print(f"Running GPT eval on experiment: {name}" )
    results = run_gpt_eval_on_experiment(exp, max_workers = 5)
    with open(f"experiment_results/gpt_evals/{name}.pkl", "wb") as f:
        pickle.dump(results, f)
    end_time = time.time()
    print(f"Completed GPT eval on experiment: {name} in {end_time - start_time:.2f} seconds")
master_end_time = time.time()
print(f"Completed all GPT evals in {master_end_time - master_start_time:.2f} seconds")