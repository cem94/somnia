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
        """Return encoding for this dataset"""
        pass

    @property
    @abstractmethod
    def char_mappings(self) -> dict:
        """Return character mappings for this dataset"""
        pass

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        content = text.strip()
        # Replace character mappings
        for key, value in self.char_mappings.items():
            content = content.replace(key, value)
        # Normalize whitespace
        return ' '.join(content.split())
    
    def split_into_chunks(self, text: str) -> list[str]:
        """
        Split text into word-based chunks approximating token limits.
        
        Note: This uses word count as approximation. For exact token count,
        a trained tokenizer would be needed.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), TokenizerConfig.MAX_SEQ_LEN):
            chunk_words = words[i:i + TokenizerConfig.MAX_SEQ_LEN]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
                
        return chunks
    
    def format_chunk_with_tokens(self, chunk: str) -> dict:
        """
        Format chunk with BOS/EOS tokens.
        
        Args:
            chunk: Text chunk to format
            
        Returns:
            Dictionary with formatted text
        """
        if not chunk.strip():
            return {"text": ""}
            
        formatted_text = (
            TokenizerConfig.BOS_TOKEN_VALUE + 
            chunk + 
            TokenizerConfig.EOS_TOKEN_VALUE
        )
        return {"text": formatted_text}
    
    @abstractmethod
    def process_file(self, input_file: str) -> list[dict]:
        """Process a single file and return list of formatted chunks"""
        pass
    
    @abstractmethod
    def process_directory(self, output_file: str) -> None:
        """Process all files in directory"""
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
            "--": " ",  # Remove double dash
        }
    
    @property
    def story_pattern(self) -> list[str]:
        """Regex patterns to remove title and metadata from stories"""
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
    
    def process_file(self, input_file: str) -> list[dict]:
        """
        Process a single fairy tale file and return list of formatted chunks.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of dictionaries with formatted text chunks
        """
        try:
            with open(input_file, 'r', encoding=self.encoding, errors='ignore') as f:
                text = f.read()
                
            # Clean and normalize text
            cleaned_text = self.clean_text(text)
            
            # Remove title and metadata patterns
            for pattern in self.story_pattern:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
            
            cleaned_text = cleaned_text.strip()
            
            if not cleaned_text:
                LOGGER.warning(f"No content after cleaning in file: {input_file}")
                return []
            
            # Split into chunks
            chunks = self.split_into_chunks(cleaned_text)
            
            # Format chunks with tokens
            formatted_chunks = [
                self.format_chunk_with_tokens(chunk) 
                for chunk in chunks
            ]
            
            return formatted_chunks
            
        except Exception as e:
            LOGGER.error(f"Error processing {input_file}: {e}")
            return []
    
    def process_directory(self, output_file: str) -> None:
        """
        Process all fairy tale files in directory.
        
        Args:
            output_file: Path to output JSONL file
        """
        if not os.path.exists(self.input_dir):
            LOGGER.warning(f"Directory not found: {self.input_dir}")
            return
            
        try:
            files = os.listdir(self.input_dir)
            processed_files = 0
            total_chunks = 0
            
            with open(output_file, 'a', encoding='utf-8') as out_f:
                for file in natsorted(files):
                    if not file.endswith('.txt'):
                        continue
                        
                    file_path = os.path.join(self.input_dir, file)
                    file_chunks = self.process_file(file_path)
                    
                    for record in file_chunks:
                        out_f.write(json.dumps(record) + '\n')
                        out_f.flush()
                    
                    processed_files += 1
                    total_chunks += len(file_chunks)
                    LOGGER.debug(f"Processed file: {file} ({len(file_chunks)} chunks)")
            
            LOGGER.info(f"FairyTalesProcessor: {processed_files} files, {total_chunks} chunks")
                            
        except Exception as e:
            LOGGER.error(f"Error processing directory {self.input_dir}: {e}")
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
            '\u201c': '"', '\u201d': '"',  # Double quotation marks
            '\u2018': "'", '\u2019': "'",  # Single quotation marks
            '\u2013': '-', '\u2014': '-',  # En dash, em dash
            '\u2026': '...',               # Ellipsis
            '\u2022': '*',                 # Bullet
            '\u00a0': ' ',                 # Non-breaking space
        }
    
    @property
    def story_separator_patterns(self) -> list[str]:
        """Regex patterns to identify story separators/titles"""
        return [
            r"(?m)^[A-Z0-9][A-Z0-9 '\-\.]{2,}$",    # Title case lines
            r"(\n\n[\w\s'\-\.]{2,50}\n\n)"          # Short lines between double newlines
        ]
        
    def split_into_stories(self, text: str) -> list[str]:
        """
        Extract individual stories by identifying content blocks.
        
        Args:
            text: Full text content from file
            
        Returns:
            List of individual story texts
        """
        content = text
        
        # Apply character mappings
        for old_char, new_char in self.char_mappings.items():
            content = content.replace(old_char, new_char)
        
        # Replace separator patterns with double newlines
        for pattern in self.story_separator_patterns:
            content = re.sub(pattern, '\n\n', content, flags=re.MULTILINE)
        
        # Split on double newlines and filter empty/short content
        stories = [s.strip() for s in content.split('\n\n')]
        # Filter out very short content that's likely just titles/metadata
        valid_stories = [s for s in stories if len(s.split()) > 10]
        
        return valid_stories
    
    def process_file(self, input_file: str) -> list[dict]:
        """
        Process a single children stories file and return list of formatted chunks.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of dictionaries with formatted text chunks
        """
        try:
            with open(input_file, 'r', encoding=self.encoding) as f:
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
            LOGGER.error(f"Error processing {input_file}: {e}")
            return []
    
    def process_directory(self, output_file: str) -> None:
        """
        Process all children story files in directory.
        
        Args:
            output_file: Path to output JSONL file
        """
        if not os.path.exists(self.input_dir):
            LOGGER.warning(f"Directory not found: {self.input_dir}")
            return
            
        try:
            files = os.listdir(self.input_dir)
            processed_files = 0
            total_chunks = 0
            
            with open(output_file, 'a', encoding='utf-8') as out_f:
                for file in natsorted(files):
                    if not file.endswith('.txt'):
                        continue
                        
                    file_path = os.path.join(self.input_dir, file)
                    file_chunks = self.process_file(file_path)
                    
                    for record in file_chunks:
                        out_f.write(json.dumps(record) + '\n')
                        out_f.flush()
                    
                    processed_files += 1
                    total_chunks += len(file_chunks)
                    LOGGER.debug(f"Processed file: {file} ({len(file_chunks)} chunks)")
            
            LOGGER.info(f"ChildrenStoriesProcessor: {processed_files} files, {total_chunks} chunks")
            
        except Exception as e:
            LOGGER.error(f"Error processing directory {self.input_dir}: {e}")
            raise


def main():
    """
    Main execution function for dataset preparation.
    
    Creates a fresh training_data.jsonl file by processing all configured
    datasets through their respective processors. Existing output file
    will be removed if present.
    """
    LOGGER.info("Starting dataset preparation")
    
    # Remove existing output file
    if os.path.exists(PROCESSED_OUTPUT_FILE):
        os.remove(PROCESSED_OUTPUT_FILE)
        LOGGER.info(f"Removed existing output file: {PROCESSED_OUTPUT_FILE}")
    
    # Initialize processors
    processors = [
        FairyTalesProcessor(os.path.join(RAW_DATA_DIR, "fairy_tales")),
        ChildrenStoriesProcessor(os.path.join(RAW_DATA_DIR, "children_stories"))
    ]
    
    total_processors = len(processors)
    
    # Process each dataset
    for i, processor in enumerate(processors, 1):
        processor_name = processor.__class__.__name__
        LOGGER.info(f"Processing dataset {i}/{total_processors}: {processor_name}")
        
        try:
            processor.process_directory(PROCESSED_OUTPUT_FILE)
        except Exception as e:
            LOGGER.error(f"Failed to process {processor_name}: {e}")
            continue
    
    LOGGER.info(f"Dataset preparation completed. Output saved to: {PROCESSED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()