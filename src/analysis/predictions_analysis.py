import pandas as pd 
import statistics
import re
import ast
from typing import Optional, Dict

def task_comparison (df1, df2, df5):
    df1_ = df1.rename(columns={"parsed_output":"parsed_run_1", "label": "label_run_1"})
    df2_ = df2.rename(columns={"parsed_output":"parsed_run_2", "label": "label_run_2"})
    df5_ = df5.rename(columns={"parsed_output":"parsed_run_3", "label": "label_run_3"})
    
    df_comparison_ = df1_[['id', '05', '01','02', "parsed_run_1", 'label_run_1']]
    
    df_comparison_ = df_comparison_.merge(df2_[['id', '05', '01','02', "parsed_run_2", "label_run_2"]], on=['id', '05', '01','02'])
    df_comparison = df_comparison_.merge(df5_[['id', '05', '01','02', "parsed_run_3", "label_run_3"]], on=['id', '05', '01','02'])

    print(df_comparison.shape)

    return df_comparison


def task_agreements(df):
    # Check if all tasks agree on the label
    task_cols = ["label_run_1", "label_run_2", "label_run_3"]
    df["All_agree"] = df[task_cols].nunique(axis=1) == 1
    # df["All_agree"] = df[task_cols].eq(df["label_run_1"], axis=0).all(axis=1)

    print("Result explanation (All_agree):")
    print("# True  - All tasks gave the same label")
    print("# False - At least one task gave a different label\n")
    print(df["All_agree"].value_counts())

    return df


def annotator_distribution(df, annotator_cols, model_parsed_cols):
    for run_col in model_parsed_cols:
        print(f"\n=== Model Run: {run_col} ===")
        for ann_col in annotator_cols:
            true_counts = df[ann_col].value_counts().sort_index()

            ann_labels = sorted(true_counts.index)
            filtered_df = df[df[run_col].isin(ann_labels)]
            pred_counts = filtered_df[run_col].value_counts().sort_index() 

            # # all_labels = sorted(set(true_counts.index).union(set(pred_counts.index)))

            true_counts = true_counts.reindex(ann_labels, fill_value=0)
            pred_counts = pred_counts.reindex(ann_labels, fill_value=0)

            true_dist = true_counts / true_counts.sum()
            pred_dist = pred_counts / pred_counts.sum()
            print(true_dist)
            print("-"*10)
            print(pred_dist)
            print("\n\n")


def extract_options(text):
    match = re.search(r"Opzioni:\s*(\[[^\]]+\])", text)
    if match:
        raw_list_str = match.group(1)
        # Replace escaped single quotes \' with normal single quotes '
        cleaned_str = raw_list_str.replace("\\'", "'")
        try:
            # Safely parse the string list into a Python list
            return ast.literal_eval(cleaned_str)
        except Exception as e:
            print(f"Error parsing list: {e}")
            return None
    return None


def find_winner(row):
    if row["parsed_output"] == row['first_option']:
        return 'first_option'
    elif row["parsed_output"] == row['second_option']:
        return 'second_option'
    elif row["parsed_output"] == row['third_option']:
        return 'third_option'
    else:
        return "no match found"  # No match found


def option_number(df):
    df['first_option'] = df['options'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    df['second_option'] = df['options'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
    df['third_option'] = df['options'].apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else None)

    return df



class PredictionsAnalysis:
    def __init__(self, output_dir, data_paths: Optional[Dict[str, str]] = None):
        self.output_dir = output_dir

        default_paths = {
            'input_file': "O-Ster dataset/preprocessed/data_on5.csv",
            'original_dataset': "O-Ster dataset/original_dataset/open_stereotypes_corpus.csv",
            'run_1': f"parsed_output/parsed_output_1.csv",
            'run_2': f"parsed_output/parsed_output_2.csv",  
            'run_3': f"parsed_output/parsed_output_3.csv",
            'classification_1': f"predictions/classifications_1.csv",
            'classification_2': f"predictions/classifications_2.csv",
            'classification_3': f"predictions/classifications_3.csv"
        }
        
        if data_paths:
            default_paths.update(data_paths)

        try:
            self.input_file = pd.read_csv(default_paths['input_file'])
            self.original_dataset = pd.read_csv(default_paths['original_dataset'])
            self.run_1 = pd.read_csv(default_paths['run_1'])
            self.run_2 = pd.read_csv(default_paths['run_2'])
            self.run_3 = pd.read_csv(default_paths['run_3'])
            
            self.classification_1 = pd.read_csv(default_paths['classification_1'])
            self.classification_2 = pd.read_csv(default_paths['classification_2'])
            self.classification_3 = pd.read_csv(default_paths['classification_3'])
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required data file not found: {e}")
        except Exception as e:
            raise Exception(f"Error loading data files: {e}")

        self.list_labels = ["label_run_1", "label_run_2", "label_run_3"]
        self.list_parsed = ["parsed_run_1", "parsed_run_2",  "parsed_run_3"]
        self.list_ann = ["01", "02", "05"]

        self.df_runs_all = task_comparison(self.run_1, self.run_2, self.run_3)


        # Initialize analysis dataframes
        self.df_agreement = None
        self.df_disagreement = None
        self.all_different_labels = None


 
    def hallucination_count(self):
        list_dfs = [self.run_1, self.run_2, self.run_3]
        avg_hall = []

        for run in list_dfs:
            tot = len(self.input_file)
            tot_run = len(run)

            hallucination = 1-(tot_run/tot)
            print(hallucination)
            avg_hall.append(hallucination)

        hallucination_percentage = f"{round(statistics.mean(avg_hall), 4)*100}%"

        coverage_percentage = len(self.df_runs_all)/len(self.input_file)*100
        
        print(f"\nAverage hallucination rate: {hallucination_percentage}")
        print(f"Coverage for analysis: {coverage_percentage:.2f}%")
        
        return hallucination_percentage, coverage_percentage


    def agreement(self):
        df = task_agreements(self.df_runs_all)
        self.df_agreement = df[df["All_agree"] == True]
        agreement_stats = {}

        for run in self.list_labels: 
            count_agreement = self.df_agreement[run].value_counts()
            percentage_agreement = count_agreement / count_agreement.sum()
           
            agreement_stats[run] = {
                'counts': count_agreement.to_dict(),
                'percentages': percentage_agreement.to_dict()
            }

            print(f"\n{run}:")
            print("Counts:", count_agreement.to_dict())
            print("Percentages:", percentage_agreement.to_dict())
            print("-" * 10)

        return self.df_agreement
    

    def disagreement(self):
        df = task_agreements(self.df_runs_all)
        self.df_disagreement = df[df["All_agree"] == False]
    
        disagreement_stats = {}

        for run in self.list_labels: 
            count_disagreement = self.df_disagreement[run].value_counts()
            percentage_disagreement = count_disagreement / count_disagreement.sum()

            disagreement_stats[run] = {
                'counts': count_disagreement.to_dict(),
                'percentages': percentage_disagreement.to_dict()
            }

            print(f"\n{run}:")
            print("Counts:", count_disagreement.to_dict())
            print("Percentages:", percentage_disagreement.to_dict())
            print("-" * 10)

        return self.df_disagreement


    def all_runs_disagree(self):
        # Define the condition using apply
        all_different_labels = self.df_runs_all[
            self.df_runs_all.apply(
                lambda row: len(set([row["label_run_1"], row["label_run_2"], row["label_run_3"]])) == 3,
                axis=1
            )
        ]

        print(len(all_different_labels))
        all_different_labels = all_different_labels.merge(self.original_dataset[["id", "tweet"]], on="id", how="left")
        self.all_different_labels = all_different_labels.drop_duplicates(subset=["id", "05","01", "02"])
        
        self.all_different_labels.to_csv(f"{self.output_dir}/all_different_labels.csv", index=False)

        print("\nDistribution analysis for complete disagreements:")
        annotator_distribution(self.all_different_labels, self.list_ann, self.list_parsed)

        return self.all_different_labels
    

    def obtain_ids_all_different(self):
        ids_different_labels = self.all_different_labels["id"].tolist()
        return ids_different_labels
    


    def winning_label(self) -> Dict[str, pd.Series]:
        # Prepare classification data with options
        classifications = {
            1: self.classification_1.copy(),
            2: self.classification_2.copy(), 
            3: self.classification_3.copy()
        }
        
        runs = {1: self.run_1, 2: self.run_2, 3: self.run_3}
        
        winner_stats = {}
        
        for run_num in [1, 2, 3]:
            print(f"\n=== Processing Run {run_num} ===")
            
            # Extract options
            classifications[run_num]['options'] = classifications[run_num]['output'].apply(extract_options)
            classifications[run_num] = option_number(classifications[run_num])
            
            # Merge with run data
            cr = classifications[run_num].merge(
                runs[run_num][["id", "05", "01", "02", "parsed_output", "label"]], 
                on=["id", "05", "01", "02"], 
                how="left"
            )
            cr = cr.drop_duplicates(subset=["id", "05", "01", "02"])
            cr = cr.dropna(subset=["parsed_output"])
            
            print(f"Shapes - Classification: {classifications[run_num].shape}, "
                  f"Run: {runs[run_num].shape}, Merged: {cr.shape}")
            
            # Find winners
            cr['winner'] = cr.apply(find_winner, axis=1)
            
            cr_filtered = cr.rename(columns={"parsed_output": f"parsed_run_{run_num}", "label": f"label_run_{run_num}"})
            cr_filtered = cr_filtered.drop_duplicates(subset=["id", f"parsed_run_{run_num}"])
            cr_filtered = self.all_different_labels[["id", f"parsed_run_{run_num}", f"label_run_{run_num}"]].merge(
                cr_filtered, 
                on=["id", f"parsed_run_{run_num}", f"label_run_{run_num}"],
                how="left"
            )
            winner_counts = cr_filtered["winner"].value_counts()

            
            winner_stats[f"run_{run_num}"] = winner_counts
            print(f"Winner distribution for run {run_num}:")
            print(winner_counts)
        
        return winner_stats



    def annotator_vs_label_distr(self):
        annotator_distribution(self.df_runs_all, self.list_ann, self.list_parsed)


    def all_agree(self):
        true_counts = self.df_runs_all['05'].value_counts().sort_index()
        df_ageement_ann05 = self.df_agreement[self.df_agreement["label_run_1"] == "ann05"]
        print(df_ageement_ann05.shape)

        pred_counts = df_ageement_ann05['parsed_run_1'].value_counts().sort_index()
        
        all_labels = sorted(set(true_counts.index).union(set(pred_counts.index)))

        true_counts = true_counts.reindex(all_labels, fill_value=0)
        pred_counts = pred_counts.reindex(all_labels, fill_value=0)

        true_dist = true_counts / true_counts.sum()
        pred_dist = pred_counts / pred_counts.sum()

        print("True distribution (ann05):")
        print(true_dist)
        print("-" * 10)
        print("Predicted distribution (when all agree on ann05):")
        print(pred_dist)

        return {'true_distribution': true_dist, 'predicted_distribution': pred_dist}
    



    def generate_full_report(self) -> Dict:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("="*60)
        print("GENERATING COMPREHENSIVE PREDICTIONS ANALYSIS REPORT")
        print("="*60)
        
        report = {}
        
        # Hallucination analysis
        print("\n1. HALLUCINATION ANALYSIS")
        print("-" * 30)
        report['hallucination'] = self.hallucination_count()
        
        # Agreement analysis  
        print("\n2. AGREEMENT ANALYSIS")
        print("-" * 30)
        report['agreement_df'] = self.agreement()
        
        # Disagreement analysis
        print("\n3. DISAGREEMENT ANALYSIS") 
        print("-" * 30)
        report['disagreement_df'] = self.disagreement()
        
        # Complete disagreement analysis
        print("\n4. COMPLETE DISAGREEMENT ANALYSIS")
        print("-" * 30)
        report['all_different_df'] = self.all_runs_disagree()
        
        # Winner analysis
        print("\n5. OPTION WINNER ANALYSIS")
        print("-" * 30)
        report['winner_stats'] = self.winning_label()
        
        # Distribution comparison
        print("\n6. ANNOTATOR VS MODEL DISTRIBUTION")
        print("-" * 30)
        self.annotator_vs_label_distr()
        
        # Agreement-specific analysis
        print("\n7. ALL-AGREE ANALYSIS")
        print("-" * 30)
        report['all_agree_stats'] = self.all_agree()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - Results saved to:", self.output_dir)
        print("="*60)
        
        return report