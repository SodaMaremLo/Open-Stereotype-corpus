import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

from typing import Dict
from collections import Counter

class CorpusAnalysis:
    """
    A class for analyzing annotated text corpus with clustering and hate speech labels.
    
    Provides statistical analysis, agent/patient analysis, cluster distribution,
    and hate speech visualization capabilities.
    
    Attributes:
        df (pd.DataFrame): Original dataset
        df_cluster (pd.DataFrame): Filtered dataset excluding 'X' cluster values, those with 1 occurrence only
        output_dir (str): Directory for saving output files
        dict_english_cl (Dict): Mapping of Italian cluster names to English translations
    """

    def __init__(self,df: pd.DataFrame, output_dir: str):
        self.df = df
        self.df_cluster = df[df["cluster_10_nome_ann02"] !='X']
        self.output_dir = output_dir

        self.dict_english_cl = {
            "cluster_5_nome_ann01":{
                'SonoParassiti': "Parasites", 
                'SonoSubdoli': "Subtle", 
                'SonoImmorali': "Immoral",
                'SonoIncompatibiliConNoi': "Incompatible", 
                'SonoProblema': "Problem"
                }, 
        "cluster_5_nome_ann02": {
            "FannoQuelloCheVoglionoSenzaContribuire": "Do not contribute",
                "SonoPericolosi": "Dangerous",
                "PeggioranoLeNostreCondizioniDiVita": "Worsen our lives",
                "HannoCulturaDiversaDallaNostra": "Different culture",
                "PortanoDegrado": "Degraded"
                }, 
        "cluster_5_nome_ann05": {
            "SonoSfruttatori": "Exploiters",
                "SonoMinaccia": "Threat",
                "RovinanoItalia": "Ruin Italy",
                "SonoTutelati": "Protected",
                "SonoEstremistiReligiosi": "Radicalized"
                }
                }


    def general_statistics(self): 
        stats = {}
        # Dataset info
        stats["n_annotations"] = self.df.shape[0]
        stats['n_texts'] = len(set(self.df["id"].tolist()))

        print("Number of annotations:", stats['n_annotations'])
        print("Number of texts:", stats['n_texts'])
        print("-" * 50)

        # Rationales
        list_parsed = self.df["annotazioni_parsate"].tolist()
        counts_parsed = Counter(list_parsed)
        parsed_to_work_on = [item for item in list_parsed if counts_parsed[item] > 1]

        stats['n_rationales'] = len(set(self.df["annotazione"].tolist()))
        stats['n_unique_parsed'] = len(set(list_parsed))
        stats['n_unique_parsed_to_annotate'] = len(set(parsed_to_work_on))

        print("Number of rationales:", stats['n_rationales'])
        print("Number of parsed stereotypes (including 1 occurrence): ", stats['n_unique_parsed'])
        print("Number of parsed stereotypes (excluding 1 occurrence): ", stats['n_unique_parsed_to_annotate'])        
        print("-" * 50) 

        # Annotator statistics
        annotator_counts = self.df["annotatore"].value_counts()
        stats['annotations_per_annotator'] = annotator_counts.to_dict()
        print("Number of annotations per annotator")
        print(annotator_counts)
        print("-" * 50)

        # Texts per annotator        
        texts_per_annotator = self.df.groupby(["annotatore", "id"]).size().reset_index(name='text_count')
        annotator_summary = texts_per_annotator.groupby("annotatore").size()
        stats['texts_per_annotator'] = annotator_summary.to_dict()
        print("Number of texts annotated by each annotator")
        print(annotator_summary)
        print("-" * 50)

        # Unique text analysis
        df_no_dup = self.df.drop_duplicates(subset=["id", "annotatore"])
        id_counts = df_no_dup["id"].value_counts()
        stats['single_occurrence_texts'] = (id_counts == 1).sum()
        print("Number of IDs that occur exactly once:", stats['single_occurrence_texts'])
        print("-" * 50)

        # Mean annotations
        count_ann_text = self.df.groupby(["annotatore", "id"])["annotazione"].size()
        stats['mean_annotations_per_text'] = count_ann_text.mean()
        stats['mean_annotations_by_annotator'] = count_ann_text.groupby("annotatore").mean().to_dict()
        print("Mean annotations for (annotatore,id):", stats['mean_annotations_per_text'])

        return stats



    def agents_patients(self):
        list_annotators = ["annotatore_01", "annotatore_02", "annotatore_03", "annotatore_04", "annotatore_05"]
        dict_agent_pat = {}

        for a in list_annotators:
            df_a = self.df[self.df["annotatore"] == a]
            print(a, "-->", df_a.shape)
            num_agents = len(df_a["agent"].dropna().tolist())
            num_patients = len(df_a["patient"].dropna().tolist())
            dict_agent_pat[a] = {"count_agent": num_agents, "count_patient":num_patients}
        print(dict_agent_pat)

        count_agent_pat = pd.DataFrame.from_dict(dict_agent_pat, orient='index').reset_index()
        count_agent_pat.rename(columns={'index': 'annotatore'}, inplace=True)

        agent_ann = self.df.groupby(["agent", "annotatore"]).size().reset_index(name='text_count')
        agent_ann = agent_ann.merge(count_agent_pat[["annotatore", "count_agent"]], on='annotatore', how='left')

        agent_ann["percentage"] = ((agent_ann["text_count"] / agent_ann["count_agent"])*100).round(2)
        agent_ann.to_csv(f"{self.output_dir}/agents.csv")


        pat_ann = self.df.groupby(["patient", "annotatore"]).size().reset_index(name='text_count')
        pat_ann = pat_ann.merge(count_agent_pat[["annotatore", "count_patient"]], on='annotatore', how='left')

        pat_ann["percentage"] = ((pat_ann["text_count"] / pat_ann["count_patient"])*100).round(2)
        pat_ann.to_csv(f"{self.output_dir}/patients.csv")

        #compute threshold
        print("AGENT threshold")
        print((agent_ann.groupby("annotatore")["percentage"].mean()))
        print()
        print("PATIENT threshold")
        print((pat_ann.groupby("annotatore")["percentage"].mean()))

        return agent_ann, pat_ann



    def groups_distribution(self, n_groups=5):

        if n_groups==10:

            print("DISTRIBUTION OF CLUSTERS ACROSS CORPUS - 10")
            print()
            print(self.df_cluster["cluster_10_nome_ann05"].value_counts())
            print()
            print(self.df_cluster["cluster_10_nome_ann01"].value_counts())
            print()
            print(self.df_cluster["cluster_10_nome_ann02"].value_counts())

        elif n_groups==5:
            print("DISTRIBUTION OF CLUSTERS ACROSS CORPUS - 5")
            print()
            print(self.df_cluster["cluster_5_nome_ann05"].value_counts())
            print()
            print(self.df_cluster["cluster_5_nome_ann01"].value_counts())
            print()
            print(self.df_cluster["cluster_5_nome_ann02"].value_counts())

        else:
            raise ValueError("n_groups must be a int representing the possible number of label groups. Accepted values are 10 or 5")
        



    def hateful_comments(self):
        df_cluster_en = self.df_cluster.copy()

        for col_name, d in self.dict_english_cl.items():
            df_cluster_en.loc[:, col_name] = self.df_cluster[col_name].map(d)
            
        df_cluster_en["hs"] = df_cluster_en["hs"].map({1:"hs", 0:"not hs"})


        # A4 landscape: width x height in inches
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11.69, 4), sharey=True)

        # Custom function
        def countplot_wrapped(ax, x_column, title, show_ylabel=False):
            sns.countplot(data=df_cluster_en, x=x_column, hue='hs', ax=ax)
            labels = [textwrap.fill(label.get_text(), width=12) for label in ax.get_xticklabels()]
            tick_positions = ax.get_xticks()
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_title(title, fontsize=10)
            
            if show_ylabel:
                ax.set_ylabel('Count', fontsize=10)
            else:
                ax.set_ylabel('')

            ax.set_xlabel('')

            # Reduce y-tick label font size
            ax.tick_params(axis='y', labelsize=10)

        # Generate each plot
        countplot_wrapped(axes[0], 'cluster_5_nome_ann01', 'Duck (annotator_01)', show_ylabel=True)
        countplot_wrapped(axes[1], 'cluster_5_nome_ann02', 'Bear (annotator_02)')
        countplot_wrapped(axes[2], 'cluster_5_nome_ann05', 'Rhino (annotator_05)')

        # Shared legend at top
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(labels),
                bbox_to_anchor=(0.5, 1.05), fontsize=10, title_fontsize=9)
        for ax in axes:
            ax.legend().remove()

        # Shared x-axis label
        # fig.text(0.5, 0.02, 'Cluster', ha='center', fontsize=10)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.output_dir}/hs_all_horizontal_clean.svg", dpi=300, bbox_inches='tight')

        return fig
    

    def generate_full_report(self) -> Dict:
        print("\n1. GENERAL STATISTICS")
        print("-"*30)
        report = {'general_stats': self.general_statistics()}
        
        print("\n2. AGENTS AND PATIENTS ANALYSIS")
        print("-"*30)
        agent_df, patient_df = self.agents_patients()
        report.update({'agent_analysis': agent_df, 'patient_analysis': patient_df})
        
        print("\n3. CLUSTER DISTRIBUTION (5 clusters)")
        print("-"*30)
        report['cluster_distribution_5'] = self.groups_distribution(n_groups=5)
        
        
        print("\n4. Running hate speech visualization...")
        try:
            report['hate_speech_figure'] = self.hateful_comments()
        except:
            report['hate_speech_figure'] = None
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - Results saved to:", self.output_dir)
        print("="*60)        
        return report