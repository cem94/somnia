"""
Module for dataset analysis.

This module analyzes raw text files from the raw data directory to compute:
  - The number of .txt files per category (folder)
  - Word frequencies per folder (both total and per document)
Additionally, it analyzes processed JSONL data.
Visualization functions generate plots for these statistics.

Usage:
    Run analyze_raw_data() or analyze_processed_data() to perform analyses.
    Or run main() to execute both analyses sequentially.
"""

import os
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
from utility.logger import LOGGER
from utility.paths import RAW_DATA_DIR, PROCESSED_OUTPUT_FILE, PLOTS_DIR


def _detect_file_encoding(file_path: str) -> str:
    """
    Detect file encoding with fallback strategy.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Detected encoding string
    """
    # Try common encodings in order of preference
    encodings_to_try = ['utf-8', 'ascii', 'latin-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # Read first 1KB to test encoding
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # If all else fails, use utf-8 with error handling
    LOGGER.warning(f"Could not detect encoding for {file_path}, using utf-8 with error handling")
    return 'utf-8'


def _read_text_file(file_path: str) -> str:
    """
    Read text file with robust encoding detection.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as string, or empty string on error
    """
    encoding = _detect_file_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            return f.read().lower()
    except Exception as e:
        LOGGER.error(f"Error reading file {file_path}: {e}")
        return ""


def _extract_words(text: str) -> list[str]:
    """
    Extract words from text using regex.
    
    Args:
        text: Input text to process
        
    Returns:
        List of words (alphanumeric only)
    """
    return re.findall(r'\b\w+\b', text)


def _analyze_raw_data() -> dict:
    """
    Analyze raw text files to compute file counts and word frequencies per folder.
    
    Returns:
        dict: A summary with the keys:
            - folders: {folder: file_count}
            - total_files: Total number of text files
            - total_words: Total number of words
            - word_freq: {folder: Counter(words)} - all word occurrences
            - file_freq: {folder: Counter(words)} - each word counted once per file
            
    Raises:
        FileNotFoundError: If raw data directory doesn't exist
    """
    LOGGER.info("Starting raw data analysis")
    
    if not os.path.exists(RAW_DATA_DIR):
        LOGGER.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")
    
    summary = {
        "folders": {},
        "total_files": 0,
        "total_words": 0,
        "word_freq": {},
        "file_freq": {}
    }
    
    # Process each folder in the raw data directory
    for folder in os.listdir(RAW_DATA_DIR):
        folder_path = os.path.join(RAW_DATA_DIR, folder)
        
        if not os.path.isdir(folder_path):
            continue
            
        # Find all .txt files in the folder
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        
        if not txt_files:
            LOGGER.warning(f"No .txt files found in folder: {folder}")
            continue
        
        # Initialize counters for this folder
        summary["folders"][folder] = len(txt_files)
        summary["total_files"] += len(txt_files)
        summary["word_freq"][folder] = Counter()
        summary["file_freq"][folder] = Counter()
        
        LOGGER.info(f"Found {len(txt_files)} text files in folder: {folder}")
        
        # Process each file in the folder
        for file in txt_files:
            file_path = os.path.join(folder_path, file)
            
            # Read and process file content
            text = _read_text_file(file_path)
            if not text:
                continue
                
            words = _extract_words(text)
            
            # Update statistics
            summary["total_words"] += len(words)
            summary["word_freq"][folder].update(words)
            
            # Count each unique word once per file for file frequency
            unique_words = set(words)
            for word in unique_words:
                summary["file_freq"][folder][word] += 1
                
        LOGGER.debug(f"Processed folder '{folder}': {len(txt_files)} files, "
                    f"{sum(summary['word_freq'][folder].values())} total words")
                
    LOGGER.info(f"Raw data analysis completed: {summary['total_files']} files, "
                f"{summary['total_words']} total words")
    return summary


def _analyze_processed_data() -> dict:
    """
    Analyze processed JSONL data to compute total word count and word frequencies.
    
    Returns:
        dict: A summary with keys:
            - total_records: Total number of records in JSONL
            - total_words: Total number of words in the processed data
            - word_freq: Counter object for word frequencies
            
    Raises:
        FileNotFoundError: If processed data file doesn't exist
    """
    LOGGER.info("Starting processed data analysis")
    
    if not os.path.exists(PROCESSED_OUTPUT_FILE):
        LOGGER.error(f"Processed data file not found: {PROCESSED_OUTPUT_FILE}")
        raise FileNotFoundError(f"Processed data file not found: {PROCESSED_OUTPUT_FILE}")
    
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
                    
                    # text_field is already a string, no need to join
                    if not isinstance(text_field, str):
                        LOGGER.warning(f"Line {line_num}: 'text' field is not a string, skipping")
                        continue
                    
                    # Process text
                    text = text_field.lower()
                    words = _extract_words(text)
                    
                    # Update statistics
                    summary["total_records"] += 1
                    summary["total_words"] += len(words)
                    summary["word_freq"].update(words)
                    
                except json.JSONDecodeError as e:
                    LOGGER.error(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    LOGGER.error(f"Error processing line {line_num}: {e}")
                    continue
                    
    except Exception as e:
        LOGGER.error(f"Error reading processed data file: {e}")
        raise

    LOGGER.info(f"Processed data analysis completed: {summary['total_records']} records, "
                f"{summary['total_words']} total words")
    return summary


def plot_raw_data(summary: dict) -> None:
    """
    Generate and save plots for raw data analysis.
    
    Creates:
        - A bar chart for the number of .txt files per folder
        - Separate bar charts for the top 10 common words (by file frequency) for each folder
        
    Args:
        summary: Analysis summary from _analyze_raw_data()
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Plot 1: Number of .txt files per folder
    if summary["folders"]:
        folder_names = list(summary["folders"].keys())
        folder_counts = list(summary["folders"].values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(folder_names, folder_counts, color="skyblue", edgecolor="navy", alpha=0.7)
        plt.xlabel("Dataset Categories", fontsize=12)
        plt.ylabel("Number of .txt Files", fontsize=12)
        plt.title("Raw Data: File Count Distribution by Category", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, folder_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        folder_plot_path = os.path.join(PLOTS_DIR, "raw_data_folder_counts.png")
        plt.tight_layout()
        plt.savefig(folder_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.info(f"Saved folder count plot to {folder_plot_path}")
    
    # Plot 2: Top 10 common words per folder (using file frequency)
    for folder, counter in summary["file_freq"].items():
        if not counter:
            LOGGER.warning(f"No words found for folder: {folder}")
            continue
            
        common_words = counter.most_common(10)
        if not common_words:
            continue
            
        words, counts = zip(*common_words)
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(words, counts, color="coral", edgecolor="darkred", alpha=0.7)
        plt.xlabel("Words", fontsize=12)
        plt.ylabel("File Frequency", fontsize=12)
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
        LOGGER.info(f"Saved common words plot for '{folder}' to {common_plot_path}")


def plot_processed_data(summary: dict) -> None:
    """
    Generate and save a bar chart for the top 10 most frequent words 
    from the processed JSONL data.
    
    Args:
        summary: Analysis summary from _analyze_processed_data()
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    word_freq = summary.get("word_freq", Counter())
    
    if not word_freq:
        LOGGER.warning("No word frequencies found in processed data")
        return
    
    most_common_words = word_freq.most_common(10)
    if not most_common_words:
        LOGGER.warning("No common words found in processed data")
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
    LOGGER.info(f"Saved processed data plot to {processed_plot_path}")


def analyze_raw_data() -> dict:
    """
    Run raw data analysis, display summary statistics, and create plots.
    
    Returns:
        dict: Analysis summary
    """
    LOGGER.info("Executing raw data analysis")
    
    try:
        raw_data_summary = _analyze_raw_data()
        
        # Display summary statistics
        print("\n" + "="*60)
        print("RAW DATA ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total files processed: {raw_data_summary['total_files']}")
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
        
    except Exception as e:
        LOGGER.error(f"Failed to analyze raw data: {e}")
        raise


def analyze_processed_data() -> dict:
    """
    Run processed data analysis, display summary statistics, and create a plot.
    
    Returns:
        dict: Analysis summary
    """
    LOGGER.info("Executing processed data analysis")
    
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
        
    except Exception as e:
        LOGGER.error(f"Failed to analyze processed data: {e}")
        raise


def main():
    """
    Main function to run comprehensive data analysis.
    
    This function orchestrates both raw and processed data analyses,
    providing complete insights into the dataset characteristics.
    """
    LOGGER.info("Starting comprehensive data analysis")
    
    try:
        # Analyze raw data
        raw_summary = analyze_raw_data()
        
        # Analyze processed data
        processed_summary = analyze_processed_data()
        
        # Final summary comparison
        print("\n" + "="*60)
        print("ANALYSIS COMPARISON")
        print("="*60)
        print(f"Raw data words: {raw_summary['total_words']:,}")
        print(f"Processed data words: {processed_summary['total_words']:,}")
        
        word_retention = (processed_summary['total_words'] / raw_summary['total_words'] * 100 
                         if raw_summary['total_words'] > 0 else 0)
        print(f"Word retention rate: {word_retention:.1f}%")
        
        LOGGER.info("Data analysis completed successfully")
        
    except Exception as e:
        LOGGER.error(f"Data analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()