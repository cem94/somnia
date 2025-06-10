"""
Module for dataset preparation.

This module processes raw text files from fairy tales and children stories
and prepares them for LLM training by:
  - Reading .txt files from specified directories
  - Cleaning and normalizing text content
  - Splitting content into chunks of max 1024 tokens
  - Outputting a JSONL file suitable for model training

The processing preserves narrative structure and linguistic richness by:
  - Applying minimal text normalization
  - Maintaining original punctuation and casing
  - Preserving stop words and word forms
  - Handling UTF-8 special characters

Usage:
    Run main() to process all datasets and generate the training file.

Source:
 - https://www.kaggle.com/discussions/getting-started/251213
"""

import os
import re
import json
from natsort import natsorted
from abc import ABC, abstractmethod
from utility.logger import LOGGER
from utility.paths import PROCESSED_OUTPUT_FILE, RAW_DATA_DIR
from model.transformer.tokenizer_config import TokenizerConfig


class AbstractDataProcessor(ABC):
    """
    Abstract base class for dataset processors.
    
    Defines the interface and common functionality for processing different
    types of text datasets into a standardized training format.
    """

    @property
    @abstractmethod
    def encoding(self) -> str:
        """Return the encoding used for reading files."""
        pass

    @property
    @abstractmethod
    def char_mappings(self) -> dict:
        """Return character mappings for text normalization."""
        pass

    def __init__(self, input_directory: str):
        self.input_directory = input_directory
        
    def clean_text(self, raw_text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            raw_text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        cleaned_content = raw_text.strip()
        # Replace character mappings
        for old_char, new_char in self.char_mappings.items():
            cleaned_content = cleaned_content.replace(old_char, new_char)
        # Normalize whitespace
        return ' '.join(cleaned_content.split())
    
    def split_into_chunks(self, text_content: str) -> list[str]:
        """
        Split text into word-based chunks approximating token limits.
        
        Note: This uses word count as approximation. For exact token count,
        a trained tokenizer would be needed.
        
        Args:
            text_content: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text_content.strip():
            return []
            
        chunks = []
        words = text_content.split()
        
        for i in range(0, len(words), TokenizerConfig.MAX_SEQ_LEN):
            chunk_words = words[i:i + TokenizerConfig.MAX_SEQ_LEN]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
                
        return chunks
    
    def format_chunk_with_tokens(self, chunk_text: str) -> dict:
        """
        Format chunk with BOS/EOS tokens.
        
        Args:
            chunk_text: Text chunk to format
            
        Returns:
            Dictionary with formatted text
        """
        if not chunk_text.strip():
            return {"text": ""}
            
        formatted_text = (
            TokenizerConfig.BOS_TOKEN_VALUE + 
            chunk_text + 
            TokenizerConfig.EOS_TOKEN_VALUE
        )
        return {"text": formatted_text}
    
    @abstractmethod
    def process_file(self, file_path: str) -> list[dict]:
        """Process a single file and return list of formatted chunks."""
        pass
    
    @abstractmethod
    def process_directory(self, output_file_path: str) -> None:
        """Process all files in the directory."""
        pass


class FairyTalesProcessor(AbstractDataProcessor):
    """
    Processor for fairy tales dataset.
    
    Handles single-story-per-file text documents, specifically formatted
    for fairy tale collections with titles and story text.
    """
    
    @property
    def encoding(self) -> str:
        return "ascii"
        
    @property
    def char_mappings(self) -> dict:
        return {
            "--": " ",  # Replace double dash with space
            "*": "",    # Remove asterisk
        }
    
    @property
    def files_to_skip(self) -> list:
        return [
            "366.txt",
            "368.txt",
            "377.txt",
            "385.txt",
            "388.txt",
            "398.txt",
            "399.txt",
            "400.txt",
            "401.txt",
            "403.txt",
            "409.txt",
            "420.txt",
            "421.txt"
            "473.txt",
            "640.txt",
            "1479.txt",
            "1480.txt",
            "1481.txt",
            "1483.txt",
            "1484.txt",
            "1485.txt",
            "1486.txt",
            "1487.txt",
            "1488.txt",
            "1489.txt",
            "1490.txt",
            "1491.txt",
            "1492.txt",
            "1493.txt",
            "1494.txt",
            "1495.txt",
            "1496.txt",
            "1497.txt",
            "1498.txt",
            "1499.txt",
            "1500.txt",
            "1501.txt",
            "1502.txt",
            "1503.txt",
            "1504.txt",
            "1505.txt",
            "1506.txt",
            "1507.txt",
            "1508.txt",
            "1509.txt",
            "1510.txt",
            "1511.txt",
            "1512.txt",
            "1513.txt",
            "1514.txt",
            "1515.txt",
            "1516.txt",
            "1517.txt",
            "1518.txt",
            "1519.txt",
            "1520.txt",
            "1521.txt",
            "1522.txt",
            "1523.txt",
            "1524.txt",
            "1525.txt",
            "1526.txt",
            "1527.txt",
        ]
    
    @property
    def story_patterns(self) -> list[str]:
        """Regex patterns to remove titles and metadata from stories."""
        return [
            r'^(?:[A-Z0-9\-\'\,\(\)\:]{2,}\s)+',
            r'^\([A-Z ]+\)',
            r'^([A-Z]+\s)+[A-Z]{2,}\s',
            r'^[^A-Za-z0-9]+',
            r'^\d+\]',
            r'^\.\s+',
            r'^[A-Z\,\: ]{3,}\b',
            r'^([a-zA-Z\._]{,4}\s[0-9]+[\.\)\]\_]+)',
            r'^([A-Za-z :\']+[\.\_\)\]]+)\s(\[[A-Za-z:_ &]+[\.\]_]+)?',
            r'^(.{,1000}\[[\d]+\](\s?))'
        ]
    
    def process_file(self, file_path: str) -> list[dict]:
        """
        Process a single fairy tale file and return list of formatted chunks.
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of dictionaries with formatted text chunks
        """
        try:
            with open(file_path, 'r', encoding=self.encoding, errors='ignore') as file:
                raw_text = file.read()
                
            # Clean and normalize text
            cleaned_text = self.clean_text(raw_text)
            
            # Remove title and metadata patterns
            for pattern in self.story_patterns:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
            
            cleaned_text = cleaned_text.strip()
            
            if not cleaned_text:
                LOGGER.warning(f"No content after cleaning in file: {file_path}")
                return []
            
            # Split into chunks
            text_chunks = self.split_into_chunks(cleaned_text)
            
            # Format chunks with tokens
            formatted_chunks = [
                self.format_chunk_with_tokens(chunk) 
                for chunk in text_chunks
            ]
            
            return formatted_chunks
            
        except Exception as error:
            LOGGER.error(f"Error processing {file_path}: {error}")
            return []
    
    def process_directory(self, output_file: str) -> None:
        """
        Process all fairy tale files in directory.
        
        Args:
            output_file: Path to output JSONL file
        """
        if not os.path.exists(self.input_directory):
            LOGGER.warning(f"Directory not found: {self.input_directory}")
            return
            
        try:
            files = os.listdir(self.input_directory)
            processed_files = 0
            total_chunks = 0
            
            with open(output_file, 'a', encoding='utf-8') as out_f:
                for file in natsorted(files):
                    if not file.endswith('.txt'):
                        continue

                    if file in self.files_to_skip:
                        continue
                        
                    file_path = os.path.join(self.input_directory, file)
                    file_chunks = self.process_file(file_path)
                    
                    for record in file_chunks:
                        out_f.write(json.dumps(record) + '\n')
                        out_f.flush()
                    
                    processed_files += 1
                    total_chunks += len(file_chunks)
                    LOGGER.debug(f"Processed file: {file} ({len(file_chunks)} chunks)")
            
            LOGGER.info(f"FairyTalesProcessor: {processed_files} files, {total_chunks} chunks")
                            
        except Exception as e:
            LOGGER.error(f"Error processing directory {self.input_directory}: {e}")
            raise


class ChildrenStoriesProcessor(AbstractDataProcessor):
    """
    Processor for children stories dataset.
    
    Handles multi-story-per-file text documents, with stories separated
    by titles and formatting patterns. Includes special character handling
    for UTF-8 encoded files.
    """
    
    @property
    def encoding(self) -> str:
        return "utf-8"
        
    @property
    def char_mappings(self) -> dict:
        return {
            '\u201c': '"', '\u201d': '"',  # Replace double quotation marks
            '\u2018': "'", '\u2019': "'",  # Replace single quotation marks
            '\u2013': '-', '\u2014': '-',  # Replace en dash and em dash
            '\u2026': '...',               # Replace ellipsis
            '\u2022': '*',                 # Replace bullet
            '\u00a0': ' ',                 # Replace non-breaking space
        }
    
    @property
    def story_separator_patterns(self) -> list[str]:
        """Regex patterns to identify story separators/titles."""
        return [
            r"(?m)^[A-Z0-9][A-Z0-9 '\-\.]{2,}$",    # Title case lines
            r"(\n\n[\w\s'\-\.]{2,50}\n\n)"          # Short lines between double newlines
        ]
        
    def split_into_stories(self, raw_text: str) -> list[str]:
        """
        Extract individual stories by identifying content blocks.
        
        Args:
            raw_text: Full text content from file
            
        Returns:
            List of individual story texts
        """
        content = raw_text
        
        # Apply character mappings
        for old_char, new_char in self.char_mappings.items():
            content = content.replace(old_char, new_char)
        
        # Replace separator patterns with double newlines
        for pattern in self.story_separator_patterns:
            content = re.sub(pattern, '\n\n', content, flags=re.MULTILINE)
        
        # Split on double newlines and filter empty/short content
        stories = [story.strip() for story in content.split('\n\n')]
        # Filter out very short content that's likely just titles/metadata
        valid_stories = [story for story in stories if len(story.split()) > 10]

        LOGGER.info(f"Found {len(valid_stories)} valid stories in children stories file.")
        
        return valid_stories
    
    def process_file(self, file_path: str) -> list[dict]:
        """
        Process a single children stories file and return list of formatted chunks.
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of dictionaries with formatted text chunks
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                text = f.read()
            
            all_chunks = []
            stories = self.split_into_stories(text)
            
            for story in stories:
                # Clean and normalize each story
                cleaned_story = self.clean_text(story)
                
                if not cleaned_story:
                    continue
                
                # Split story into chunks
                story_chunks = self.split_into_chunks(cleaned_story)
                
                # Format chunks with tokens
                formatted_chunks = [
                    self.format_chunk_with_tokens(chunk) 
                    for chunk in story_chunks
                ]
                
                all_chunks.extend(formatted_chunks)
            
            return all_chunks
            
        except Exception as e:
            LOGGER.error(f"Error processing {file_path}: {e}")
            return []
    
    def process_directory(self, output_file: str) -> None:
        """
        Process all children story files in directory.
        
        Args:
            output_file: Path to output JSONL file
        """
        if not os.path.exists(self.input_directory):
            LOGGER.warning(f"Directory not found: {self.input_directory}")
            return
            
        try:
            files = os.listdir(self.input_directory)
            processed_files = 0
            total_chunks = 0
            
            with open(output_file, 'a', encoding='utf-8') as out_f:
                for file in natsorted(files):
                    if not file.endswith('.txt'):
                        continue
                        
                    file_path = os.path.join(self.input_directory, file)
                    file_chunks = self.process_file(file_path)
                    
                    for record in file_chunks:
                        out_f.write(json.dumps(record) + '\n')
                        out_f.flush()
                    
                    processed_files += 1
                    total_chunks += len(file_chunks)
                    LOGGER.debug(f"Processed file: {file} ({len(file_chunks)} chunks)")
            
            LOGGER.info(f"ChildrenStoriesProcessor: {processed_files} files, {total_chunks} chunks")
            
        except Exception as e:
            LOGGER.error(f"Error processing directory {self.input_directory}: {e}")
            raise


def main():
    """
    Main execution function for dataset preparation.
    
    Creates a fresh training_data.jsonl file by processing all configured
    datasets through their respective processors. Existing output file
    will be removed if present.
    """
    LOGGER.info("Pipeline Stage 1: Starting dataset preparation.")
    
    # Remove existing output file
    if os.path.exists(PROCESSED_OUTPUT_FILE):
        os.remove(PROCESSED_OUTPUT_FILE)
        LOGGER.info(f"Existing output file removed: {PROCESSED_OUTPUT_FILE}.")
    
    # Initialize processors
    processors = [
        FairyTalesProcessor(os.path.join(RAW_DATA_DIR, "fairy_tales")),
        ChildrenStoriesProcessor(os.path.join(RAW_DATA_DIR, "children_stories"))
    ]
    
    total_processors = len(processors)
    LOGGER.info(f"Initialized {total_processors} dataset processors.")
    
    # Process each dataset
    for i, processor in enumerate(processors, 1):
        processor_name = processor.__class__.__name__
        LOGGER.info(f"Processing dataset {i}/{total_processors}: {processor_name}.")
        
        try:
            processor.process_directory(PROCESSED_OUTPUT_FILE)
            LOGGER.info(f"Dataset {processor_name} processed successfully.")
        except Exception as error:
            LOGGER.error(f"Failed to process {processor_name}: {error}.")
            continue
    
    LOGGER.info(f"Pipeline Stage 1 completed. Output saved to: {PROCESSED_OUTPUT_FILE}.")

if __name__ == "__main__":
    main()