import os
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_luo_dataset() -> pd.DataFrame:
    """
    Load the Luo dataset from CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Get the directory where this script is located
    base_dir = os.path.dirname(__file__)
    
    # Correct dataset path inside "data" folder
    dataset_path = os.path.join(base_dir, "data", "luo_dataset.csv")
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Load CSV
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset with {len(df)} rows")
    return df

def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate the dataset for required columns and data quality.
    
    Args:
        df: Input dataframe
        
    Returns:
        bool: True if dataset is valid
    """
    required_columns = ['en', 'luo']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for empty values
    empty_en = df['en'].isna().sum()
    empty_luo = df['luo'].isna().sum()
    
    if empty_en > 0 or empty_luo > 0:
        logger.warning(f"Found empty values - English: {empty_en}, Luo: {empty_luo}")
        # Remove rows with empty values
        df.dropna(subset=['en', 'luo'], inplace=True)
        logger.info(f"Removed rows with empty values. Remaining: {len(df)} rows")
    
    # Check for very short or very long texts
    short_en = (df['en'].str.len() < 3).sum()
    short_luo = (df['luo'].str.len() < 3).sum()
    long_en = (df['en'].str.len() > 500).sum()
    long_luo = (df['luo'].str.len() > 500).sum()
    
    if short_en > 0 or short_luo > 0:
        logger.warning(f"Found very short texts - English: {short_en}, Luo: {short_luo}")
    
    if long_en > 0 or long_luo > 0:
        logger.warning(f"Found very long texts - English: {long_en}, Luo: {long_luo}")
    
    logger.info("Dataset validation completed")
    return True

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # remove extra spaces
    return text

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by cleaning text and adding metadata.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    logger.info("Starting dataset preprocessing...")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Clean text
    df_processed["en_clean"] = df_processed["en"].apply(clean_text)
    df_processed["luo_clean"] = df_processed["luo"].apply(clean_text)
    
    # Add text length features
    df_processed["en_length"] = df_processed["en_clean"].str.len()
    df_processed["luo_length"] = df_processed["luo_clean"].str.len()
    
    # Add word count features
    df_processed["en_word_count"] = df_processed["en_clean"].str.split().str.len()
    df_processed["luo_word_count"] = df_processed["luo_clean"].str.split().str.len()
    
    # Filter out very short or very long texts
    df_processed = df_processed[
        (df_processed["en_length"] >= 3) & 
        (df_processed["luo_length"] >= 3) &
        (df_processed["en_length"] <= 500) & 
        (df_processed["luo_length"] <= 500)
    ]
    
    logger.info(f"Preprocessing completed. Final dataset size: {len(df_processed)} rows")
    return df_processed

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: Input dataframe
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting dataset into train/validation/test sets...")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Second split: separate validation from train
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size), random_state=random_state, shuffle=True
    )
    
    logger.info(f"Dataset split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def create_vocabulary(df: pd.DataFrame, min_freq: int = 2) -> Dict[str, Dict]:
    """
    Create vocabulary from the dataset.
    
    Args:
        df: Input dataframe
        min_freq: Minimum frequency for a word to be included in vocabulary
        
    Returns:
        Dict containing English and Luo vocabularies
    """
    logger.info("Creating vocabulary...")
    
    # Collect all words
    en_words = []
    luo_words = []
    
    for text in df["en_clean"]:
        en_words.extend(text.split())
    
    for text in df["luo_clean"]:
        luo_words.extend(text.split())
    
    # Create word frequency dictionaries
    en_freq = {}
    luo_freq = {}
    
    for word in en_words:
        en_freq[word] = en_freq.get(word, 0) + 1
    
    for word in luo_words:
        luo_freq[word] = luo_freq.get(word, 0) + 1
    
    # Filter by minimum frequency
    en_vocab = {word: freq for word, freq in en_freq.items() if freq >= min_freq}
    luo_vocab = {word: freq for word, freq in luo_freq.items() if freq >= min_freq}
    
    # Add special tokens
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    
    en_vocab_with_ids = {token: i for i, token in enumerate(special_tokens)}
    luo_vocab_with_ids = {token: i for i, token in enumerate(special_tokens)}
    
    # Add regular words
    for i, word in enumerate(en_vocab.keys()):
        en_vocab_with_ids[word] = i + len(special_tokens)
    
    for i, word in enumerate(luo_vocab.keys()):
        luo_vocab_with_ids[word] = i + len(special_tokens)
    
    vocabulary = {
        'en': {
            'word2idx': en_vocab_with_ids,
            'idx2word': {idx: word for word, idx in en_vocab_with_ids.items()},
            'freq': en_freq,
            'size': len(en_vocab_with_ids)
        },
        'luo': {
            'word2idx': luo_vocab_with_ids,
            'idx2word': {idx: word for word, idx in luo_vocab_with_ids.items()},
            'freq': luo_freq,
            'size': len(luo_vocab_with_ids)
        }
    }
    
    logger.info(f"Vocabulary created - English: {vocabulary['en']['size']} words, Luo: {vocabulary['luo']['size']} words")
    return vocabulary

def save_processed_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                       vocabulary: Dict, output_dir: str = None) -> None:
    """
    Save processed datasets and vocabulary.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        vocabulary: Vocabulary dictionary
        output_dir: Output directory (defaults to data folder)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, "vocabulary.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    
    # Save dataset statistics
    stats = {
        'train_size': len(train_df),
        'validation_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(train_df) + len(val_df) + len(test_df),
        'en_vocab_size': vocabulary['en']['size'],
        'luo_vocab_size': vocabulary['luo']['size']
    }
    
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved processed data to: {output_dir}")
    logger.info(f"Dataset statistics: {stats}")

def main():
    """
    Main function to run the complete data preparation pipeline.
    """
    try:
        # Load dataset
        dataset = load_luo_dataset()
        
        # Validate dataset
        if not validate_dataset(dataset):
            logger.error("Dataset validation failed")
            return
        
        # Preprocess dataset
        dataset = preprocess_dataset(dataset)
        
        # Split dataset
        train_df, val_df, test_df = split_dataset(dataset)
        
        # Create vocabulary
        vocabulary = create_vocabulary(dataset)
        
        # Save processed data
        save_processed_data(train_df, val_df, test_df, vocabulary)
        
        # Print sample
        print("\n✅ Preprocessed dataset sample:")
        print(dataset.head())
        
        print(f"\n📊 Dataset Statistics:")
        print(f"Total samples: {len(dataset)}")
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        print(f"English vocabulary size: {vocabulary['en']['size']}")
        print(f"Luo vocabulary size: {vocabulary['luo']['size']}")
        
        # Save cleaned version in the same data folder
        out_path = os.path.join(os.path.dirname(__file__), "data", "luo_dataset_clean.csv")
        dataset.to_csv(out_path, index=False)
        print(f"\n💾 Saved cleaned dataset to: {out_path}")
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
