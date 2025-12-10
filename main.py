import pandas as pd 
import argparse
from src.data.preprocessing import cluster_based_filter
from src.model.classification import classify
from src.model.parse_output import parse_single_file
from src.analysis.corpus_analysis import CorpusAnalysis
from src.analysis.predictions_analysis import PredictionsAnalysis

def main():
    parser = argparse.ArgumentParser(description="Run O-Ster pipeline.")

    parser.add_argument(
        "--classify", action="store_true",
        help="Run classification on the filtered dataset"
    )
    parser.add_argument(
        "--parse", action="store_true",
        help="Parse the classification output file"
    )
    parser.add_argument(
        "--corpus-analysis", action="store_true",
        help="Run analysis on the original corpus"
    )
    parser.add_argument(
        "--prediction-analysis", action="store_true",
        help="Run analysis on the predictions"
    )

    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv("O-Ster dataset/original_dataset/open_stereotypes_corpus.csv", sep=",")
    df_on5 = cluster_based_filter(df)
    print("\n\n")
    
    if args.classify:
        print("Running classification...")
        classify(
            df=df_on5,
            pred_path="./predictions",
            processed_fileame="processed_dataset_1",
            prediction_filename="classifications_1",
            seed=42
        )

    if args.parse:
        print("Parsing classification output...")
        parse_single_file(
            input_file="./predictions/classifications_1.csv",
            output_file="./parsed_output/parsed_output_1.csv"
        )

    if args.corpus_analysis:
        print("Running corpus analysis...")
        corpus_analyzer = CorpusAnalysis(df=df, output_dir="./analysis_output")
        corpus_analyzer.generate_full_report()
        print("Corpus analysis completed.")

    if args.prediction_analysis:
        print("Running prediction analysis...")
        prediction_analyzer = PredictionsAnalysis(output_dir="./analysis_output")
        prediction_analyzer.generate_full_report()
        print("Prediction analysis completed.")

if __name__ == "__main__":
    main()