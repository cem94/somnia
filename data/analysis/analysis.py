"""
Module for exploratory data analysis.

This module provides functionality to analyze the processed dataset,
generate statistics, and visualize key insights to ensure data quality
and readiness for model training.

Features:
  - Compute basic statistics (e.g., token counts, chunk distribution)
  - Generate visualizations (e.g., histograms, word clouds)
  - Log insights for debugging and optimization

Usage:
    Run main() to perform analysis on the processed dataset.
"""

import os
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
from utility.logger import LOGGER
from utility.paths import RAW_DATA_DIR, PROCESSED_OUTPUT_FILE, PLOTS_DIR
from data.processed.prepare_dataset import ChildrenStoriesProcessor


def _detect_file_encoding(file_path: str) -> str:
    """
    Detect file encoding with fallback strategy.
    
    Args:
        file_path: Path to the file to analyze.
        
    Returns:
        Detected encoding string.
    """
    LOGGER.debug(f"Detecting encoding for file: {file_path}.")
    encodings_to_try = ['ascii']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(512)  # Read first 512 bytes to test encoding.
            LOGGER.debug(f"Detected encoding '{encoding}' for file: {file_path}.")
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    LOGGER.warning(f"Could not detect encoding for {file_path}. Using utf-8 with error handling.")
    return 'utf-8'


def _read_text_file(file_path: str) -> str:
    """
    Read text file with robust encoding detection.
    
    Args:
        file_path: Path to the text file.
        
    Returns:
        File content as string, or empty string on error.
    """
    encoding = _detect_file_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read().lower()
            LOGGER.debug(f"Successfully read file: {file_path}.")
            return content
    except Exception as error:
        LOGGER.error(f"Error reading file {file_path}: {error}.")
        return ""


def _extract_words(text: str) -> list[str]:
    """
    Extract words from text using regex.
    
    Args:
        text: Input text to process.
        
    Returns:
        List of words (alphanumeric only).
    """
    return re.findall(r'\b\w+\b', text)


def _analyze_raw_data() -> dict:
    """
    Analyze raw text files to compute story counts and word frequencies per folder.
    
    Returns:
        dict: A summary with the keys:
            - folders: {folder: story_count}
            - total_stories: Total number of stories.
            - total_words: Total number of words.
            - word_freq: {folder: Counter(words)} - all word occurrences.
    """
    LOGGER.info("Pipeline Stage 1: Starting raw data analysis.")
    
    if not os.path.exists(RAW_DATA_DIR):
        LOGGER.error(f"Raw data directory not found: {RAW_DATA_DIR}.")
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}.")
    
    summary = {
        "folders": {},
        "total_stories": 0,
        "total_words": 0,
        "word_freq": {}
    }
    
    for folder in os.listdir(RAW_DATA_DIR):
        folder_path = os.path.join(RAW_DATA_DIR, folder)
        
        if not os.path.isdir(folder_path):
            LOGGER.warning(f"Skipping non-directory item: {folder}.")
            continue
        
        # Initialize counters for the folder
        summary["folders"][folder] = 0
        summary["word_freq"][folder] = Counter()
        
        if "children_stories" in folder_path:
            # Process children stories dataset
            try:
                # Find .txt files in the directory instead of hardcoding filename
                txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
                
                if not txt_files:
                    LOGGER.warning(f"No .txt files found in folder: {folder}.")
                    continue
                
                # Use the first .txt file found (assuming single file dataset)
                file_path = os.path.join(folder_path, txt_files[0])

                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                processor = ChildrenStoriesProcessor("")
                stories = processor.split_into_stories(text)
                summary["folders"][folder] += len(stories)
                summary["total_stories"] += len(stories)
                
                for story in stories:
                    words = _extract_words(story)
                    summary["total_words"] += len(words)
                    summary["word_freq"][folder].update(words)
                
                LOGGER.info(f"Processed {len(stories)} stories in folder: {folder}.")
            except Exception as error:
                LOGGER.error(f"Error processing children stories in folder {folder}: {error}.")
                continue
        
        elif "fairy_tales" in folder_path:
            # Process fairy tales dataset
            txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
            
            if not txt_files:
                LOGGER.warning(f"No .txt files found in folder: {folder}.")
                continue
            
            LOGGER.info(f"Found {len(txt_files)} text files in folder: {folder}.")
            
            for file in txt_files:
                file_path = os.path.join(folder_path, file)
                try:
                    text = _read_text_file(file_path)
                    if not text:
                        continue
                    
                    summary["folders"][folder] += 1
                    summary["total_stories"] += 1
                    
                    words = _extract_words(text)
                    summary["total_words"] += len(words)
                    summary["word_freq"][folder].update(words)
                except Exception as error:
                    LOGGER.error(f"Error processing file {file_path}: {error}.")
                    continue
            
            LOGGER.info(f"Processed {len(txt_files)} fairy tales in folder: {folder}.")
        
        else:
            LOGGER.warning(f"Unknown dataset type in folder: {folder}. Skipping.")
    
    LOGGER.info(f"Pipeline Stage 1 completed: {summary['total_stories']} stories, "
                f"{summary['total_words']} total words.")
    return summary


def _analyze_processed_data() -> dict:
    """
    Analyze processed JSONL data to compute total word count and word frequencies.
    
    Returns:
        dict: A summary with keys:
            - total_records: Total number of records in JSONL.
            - total_words: Total number of words in the processed data.
            - word_freq: Counter object for word frequencies.
    """
    LOGGER.info("Pipeline Stage 2: Starting processed data analysis.")
    
    if not os.path.exists(PROCESSED_OUTPUT_FILE):
        LOGGER.error(f"Processed data file not found: {PROCESSED_OUTPUT_FILE}.")
        raise FileNotFoundError(f"Processed data file not found: {PROCESSED_OUTPUT_FILE}.")
    
    summary = {
        "total_records": 0,
        "total_words": 0,
        "word_freq": Counter()
    }
    
    try:
        with open(PROCESSED_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = json.loads(line)
                    text_field = record.get("text", "")
                    
                    if not isinstance(text_field, str):
                        LOGGER.warning(f"Line {line_num}: 'text' field is not a string. Skipping.")
                        continue
                    
                    text = text_field.lower()
                    words = _extract_words(text)
                    
                    summary["total_records"] += 1
                    summary["total_words"] += len(words)
                    summary["word_freq"].update(words)
                    
                except json.JSONDecodeError as error:
                    LOGGER.error(f"Invalid JSON on line {line_num}: {error}.")
                    continue
                except Exception as error:
                    LOGGER.error(f"Error processing line {line_num}: {error}.")
                    continue
                    
    except Exception as error:
        LOGGER.error(f"Error reading processed data file: {error}.")
        raise

    LOGGER.info(f"Pipeline Stage 2 completed: {summary['total_records']} records, "
                f"{summary['total_words']} total words.")
    return summary


def plot_raw_data(summary: dict) -> None:
    """
    Generate and save plots for raw data analysis.
    
    Creates:
        - A bar chart for the number of stories per folder.
        - Separate bar charts for the top 10 common words (by word frequency) for each folder.
        
    Args:
        summary: Analysis summary from _analyze_raw_data()
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Plot 1: Number of stories per folder
    if summary["folders"]:
        folder_names = list(summary["folders"].keys())
        story_counts = list(summary["folders"].values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(folder_names, story_counts, color="skyblue", edgecolor="navy", alpha=0.7)
        plt.xlabel("Dataset Categories", fontsize=12)
        plt.ylabel("Number of Stories", fontsize=12)
        plt.title("Raw Data: Story Count Distribution by Category", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, story_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                     str(count), ha='center', va='bottom', fontweight='bold')
        
        story_plot_path = os.path.join(PLOTS_DIR, "raw_data_story_counts.png")
        plt.tight_layout()
        plt.savefig(story_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.info(f"Saved story count plot to {story_plot_path}.")
    
    # Plot 2: Top 10 common words per folder (using word frequency)
    for folder, counter in summary["word_freq"].items():
        if not counter:
            LOGGER.warning(f"No words found for folder: {folder}.")
            continue
            
        common_words = counter.most_common(10)
        if not common_words:
            continue
            
        words, counts = zip(*common_words)
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(words, counts, color="coral", edgecolor="darkred", alpha=0.7)
        plt.xlabel("Words", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Raw Data: Top 10 Common Words - {folder.replace('_', ' ').title()}", 
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                     str(count), ha='center', va='bottom', fontsize=10)
        
        safe_folder_name = folder.replace(' ', '_').replace('/', '_')
        common_plot_path = os.path.join(PLOTS_DIR, f"raw_data_{safe_folder_name}_common_words.png")
        plt.tight_layout()
        plt.savefig(common_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.info(f"Saved common words plot for '{folder}' to {common_plot_path}.")


def plot_processed_data(summary: dict) -> None:
    """
    Generate and save a bar chart for the top 10 most frequent words 
    from the processed JSONL data.
    
    Args:
        summary: Analysis summary from _analyze_processed_data().
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    word_freq = summary.get("word_freq", Counter())
    
    if not word_freq:
        LOGGER.warning("No word frequencies found in processed data.")
        return
    
    most_common_words = word_freq.most_common(10)
    if not most_common_words:
        LOGGER.warning("No common words found in processed data.")
        return
        
    words, counts = zip(*most_common_words)
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(words, counts, color="lightgreen", edgecolor="darkgreen", alpha=0.7)
    plt.xlabel("Words", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Processed Data: Top 10 Most Frequent Words", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01, 
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    processed_plot_path = os.path.join(PLOTS_DIR, "processed_data_top_words.png")
    plt.tight_layout()
    plt.savefig(processed_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"Saved processed data plot to {processed_plot_path}.")


def analyze_raw_data() -> dict:
    """
    Run raw data analysis, display summary statistics, and create plots.
    
    Returns:
        dict: Analysis summary.
    """
    LOGGER.info("Executing raw data analysis.")
    
    try:
        raw_data_summary = _analyze_raw_data()
        
        # Display summary statistics
        print("\n" + "="*60)
        print("RAW DATA ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total files processed: {raw_data_summary['total_stories']}")
        print(f"Total words found: {raw_data_summary['total_words']:,}")
        print(f"Folders analyzed: {len(raw_data_summary['folders'])}")
        
        for folder, file_count in raw_data_summary["folders"].items():
            total_words_in_folder = sum(raw_data_summary["word_freq"][folder].values())
            print(f"\n{folder.replace('_', ' ').title()}:")
            print(f"  Files: {file_count}")
            print(f"  Total words: {total_words_in_folder:,}")
            print(f"  Unique words: {len(raw_data_summary['word_freq'][folder])}")
            print(f"  Top 5 words: {raw_data_summary['word_freq'][folder].most_common(5)}")
        
        # Generate plots
        plot_raw_data(raw_data_summary)
        
        return raw_data_summary
        
    except Exception as error:
        LOGGER.error(f"Failed to analyze raw data: {error}.")
        raise


def analyze_processed_data() -> dict:
    """
    Run processed data analysis, display summary statistics, and create a plot.
    
    Returns:
        dict: Analysis summary.
    """
    LOGGER.info("Executing processed data analysis.")
    
    try:
        processed_summary = _analyze_processed_data()
        
        # Display summary statistics
        print("\n" + "="*60)
        print("PROCESSED DATA ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total records: {processed_summary['total_records']}")
        print(f"Total words: {processed_summary['total_words']:,}")
        print(f"Unique words: {len(processed_summary['word_freq'])}")
        print(f"Top 10 words: {processed_summary['word_freq'].most_common(10)}")
        
        # Generate plot
        plot_processed_data(processed_summary)
        
        return processed_summary
        
    except Exception as error:
        LOGGER.error(f"Failed to analyze processed data: {error}.")
        raise


def main():
    """
    Main function to run comprehensive data analysis.
    
    This function orchestrates both raw and processed data analyses,
    providing complete insights into the dataset characteristics.
    """
    LOGGER.info("Starting comprehensive data analysis.")
    
    try:
        analyze_raw_data()
        analyze_processed_data()
        
        LOGGER.info("Comprehensive data analysis completed successfully.")
        
    except Exception as error:
        LOGGER.error(f"Comprehensive data analysis failed: {error}.")
        raise


if __name__ == "__main__":
    main()