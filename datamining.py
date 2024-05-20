import pm4py
import pandas as pd

def read_event_log(filepath):
    event_log = pm4py.read_xes(filepath)
    return event_log

#-------------------------------------------------------------------------------------------------------------------------------------
log = read_event_log("data/BPI Challenge 2017.xes")

#-------------------------------------------------------------------------------------------------------------------------------------

trace_structure = log[log['case:concept:name'] == 'Application_303923658']
event_structure = log.columns.tolist()

print("Trace structure: {}\nEvent Structure: {}".format(trace_structure,event_structure)) ## kathe grammi  tou Dataframe einai ena event

#-------------------------------------------------------------------------------------------------------------------------------------

unique_trace_count = len(log.groupby('case:concept:name').size())

print("Number of unique traces:", unique_trace_count)

#-------------------------------------------------------------------------------------------------------------------------------------

print("Number of unique events:", len(log))

#-------------------------------------------------------------------------------------------------------------------------------------

print(log)

#-------------------------------------------------------------------------------------------------------------------------------------

filtered_event_log = pm4py.filter_time_range(log, "2016-01-01 00:00:00", "2016-01-01 23:59:59", mode='traces_intersecting')

#-------------------------------------------------------------------------------------------------------------------------------------

alpha_miner_original_net, alpha_miner_original_initial_marking, alpha_miner_original_final_marking = pm4py.discover_petri_net_alpha(log)
heuristics_miner_original_net, heuristics_miner_original_initial_marking, heuristics_miner_original_final_marking = pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.99)
inductive_miner_original_net, inductive_miner_original_initial_marking, inductive_miner_original_final_marking = pm4py.discover_petri_net_inductive(log)

alpha_miner_filtered_net, alpha_miner_filtered_initial_marking, alpha_miner_filtered_final_marking = pm4py.discover_petri_net_alpha(filtered_event_log)
heuristics_miner_filtered_net, heuristics_miner_filtered_initial_marking, heuristics_miner_filtered_final_marking = pm4py.discover_petri_net_heuristics(filtered_event_log, dependency_threshold=0.99)
inductive_miner_filtered_net, inductive_miner_filtered_initial_marking, inductive_miner_filtered_final_marking = pm4py.discover_petri_net_inductive(filtered_event_log)


#region Optional
pm4py.view_petri_net(inductive_miner_original_net, inductive_miner_original_initial_marking, inductive_miner_original_final_marking)
#endregion

#-------------------------------------------------------------------------------------------------------------------------------------

from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

# Evaluate fitness, precision, generalization, and simplicity for each model
evaluation_results = []

# Original event log with Alpha Miner model
alpha_miner_original_fitness = pm4py.fitness_token_based_replay(log, alpha_miner_original_net, alpha_miner_original_initial_marking, alpha_miner_original_final_marking)
alpha_miner_original_precision = pm4py.precision_token_based_replay(log, alpha_miner_original_net, alpha_miner_original_initial_marking, alpha_miner_original_final_marking)
alpha_miner_original_generalization = generalization_evaluator.apply(log, alpha_miner_original_net, alpha_miner_original_initial_marking, alpha_miner_original_final_marking)
alpha_miner_original_simplicity = simplicity_evaluator.apply(alpha_miner_original_net)

evaluation_results.append(("Alpha Miner (Original)", alpha_miner_original_fitness['average_trace_fitness'], alpha_miner_original_fitness['percentage_of_fitting_traces'], alpha_miner_original_precision, alpha_miner_original_generalization, alpha_miner_original_simplicity))

# Original event log with Heuristics Miner model
heuristics_miner_original_fitness = pm4py.fitness_token_based_replay(log, heuristics_miner_original_net, heuristics_miner_original_initial_marking, heuristics_miner_original_final_marking)
heuristics_miner_original_precision = pm4py.precision_token_based_replay(log, heuristics_miner_original_net, heuristics_miner_original_initial_marking, heuristics_miner_original_final_marking)
heuristics_miner_original_generalization = generalization_evaluator.apply(log, heuristics_miner_original_net, heuristics_miner_original_initial_marking, heuristics_miner_original_final_marking)
heuristics_miner_original_simplicity = simplicity_evaluator.apply(heuristics_miner_original_net)

evaluation_results.append(("Heuristics Miner (Original)", heuristics_miner_original_fitness['average_trace_fitness'], heuristics_miner_original_fitness['percentage_of_fitting_traces'], heuristics_miner_original_precision, heuristics_miner_original_generalization, heuristics_miner_original_simplicity))

# Original event log with Inductive Miner model
inductive_miner_original_fitness = pm4py.fitness_token_based_replay(log, inductive_miner_original_net, inductive_miner_original_initial_marking, inductive_miner_original_final_marking)
inductive_miner_original_precision = pm4py.precision_token_based_replay(log, inductive_miner_original_net, inductive_miner_original_initial_marking, inductive_miner_original_final_marking)
inductive_miner_original_generalization = generalization_evaluator.apply(log, inductive_miner_original_net, inductive_miner_original_initial_marking, inductive_miner_original_final_marking)
inductive_miner_original_simplicity = simplicity_evaluator.apply(inductive_miner_original_net)

evaluation_results.append(("Inductive Miner (Original)", inductive_miner_original_fitness['average_trace_fitness'], inductive_miner_original_fitness['percentage_of_fitting_traces'], inductive_miner_original_precision, inductive_miner_original_generalization, inductive_miner_original_simplicity))

# Filtered event log with Alpha Miner model
alpha_miner_filtered_fitness = pm4py.fitness_token_based_replay(filtered_event_log, alpha_miner_filtered_net, alpha_miner_filtered_initial_marking, alpha_miner_filtered_final_marking)
alpha_miner_filtered_precision = pm4py.precision_token_based_replay(filtered_event_log, alpha_miner_filtered_net, alpha_miner_filtered_initial_marking, alpha_miner_filtered_final_marking)
alpha_miner_filtered_generalization = generalization_evaluator.apply(filtered_event_log, alpha_miner_filtered_net, alpha_miner_filtered_initial_marking, alpha_miner_filtered_final_marking)
alpha_miner_filtered_simplicity = simplicity_evaluator.apply(alpha_miner_filtered_net)

evaluation_results.append(("Alpha Miner (Filtered)", alpha_miner_filtered_fitness['average_trace_fitness'], alpha_miner_filtered_fitness['percentage_of_fitting_traces'], alpha_miner_filtered_precision, alpha_miner_filtered_generalization, alpha_miner_filtered_simplicity))

# Filtered event log with Heuristics Miner model
heuristics_miner_filtered_fitness = pm4py.fitness_token_based_replay(filtered_event_log, heuristics_miner_filtered_net, heuristics_miner_filtered_initial_marking, heuristics_miner_filtered_final_marking)
heuristics_miner_filtered_precision = pm4py.precision_token_based_replay(filtered_event_log, heuristics_miner_filtered_net, heuristics_miner_filtered_initial_marking, heuristics_miner_filtered_final_marking)
heuristics_miner_filtered_generalization = generalization_evaluator.apply(filtered_event_log, heuristics_miner_filtered_net, heuristics_miner_filtered_initial_marking, heuristics_miner_filtered_final_marking)
heuristics_miner_filtered_simplicity = simplicity_evaluator.apply(heuristics_miner_filtered_net)

evaluation_results.append(("Heuristics Miner (Filtered)", heuristics_miner_filtered_fitness['average_trace_fitness'], heuristics_miner_filtered_fitness['percentage_of_fitting_traces'], heuristics_miner_filtered_precision, heuristics_miner_filtered_generalization, heuristics_miner_filtered_simplicity))

# Filtered event log with Inductive Miner model
inductive_miner_filtered_fitness = pm4py.fitness_token_based_replay(filtered_event_log, inductive_miner_filtered_net, inductive_miner_filtered_initial_marking, inductive_miner_filtered_final_marking)
inductive_miner_filtered_precision = pm4py.precision_token_based_replay(filtered_event_log, inductive_miner_filtered_net, inductive_miner_filtered_initial_marking, inductive_miner_filtered_final_marking)
inductive_miner_filtered_generalization = generalization_evaluator.apply(filtered_event_log, inductive_miner_filtered_net, inductive_miner_filtered_initial_marking, inductive_miner_filtered_final_marking)
inductive_miner_filtered_simplicity = simplicity_evaluator.apply(inductive_miner_filtered_net)

evaluation_results.append(("Inductive Miner (Filtered)", inductive_miner_filtered_fitness['average_trace_fitness'], inductive_miner_filtered_fitness['percentage_of_fitting_traces'], inductive_miner_filtered_precision, inductive_miner_filtered_generalization, inductive_miner_filtered_simplicity))

# Create a DataFrame for the evaluation results
df = pd.DataFrame(evaluation_results, columns=["Algorithm", "Avg Fitness", "Percentage fitting traces", "Precision", "Generalization", "Simplicity"])

# Display the DataFrame
print("Evaluation results:")
print(df.to_string())

