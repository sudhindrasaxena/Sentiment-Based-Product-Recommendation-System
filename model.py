# Importing Libraries
import pandas as pd
import re, nltk, spacy
import pickle as pk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import os
from typing import List, Union, Optional

class ModelError(Exception):
    """Base exception class for model-related errors."""
    pass

class DataNotFoundError(ModelError):
    """Raised when required data files are missing."""
    pass

class ModelNotLoadedError(ModelError):
    """Raised when required model artifacts cannot be loaded."""
    pass

class UserNotFoundError(ModelError):
    """Raised when a requested user is not found in the recommendation matrix."""
    pass

# NOTE: avoid downloading NLTK data at import time. If the resources are missing,
# functions below will attempt to raise a clear error or the project setup should
# download NLTK data during installation (see README).

# Lazy-loaded model artifacts (set to None until first use)
count_vector = None
tfidf_transformer = None
model = None
recommend_matrix = None

# Load spaCy language model with disabled components for efficiency. If not
# installed, the import below will raise; that's OK but caught later if needed.
try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
except Exception:
    # keep nlp as None if model isn't available; functions that need it should
    # handle this case or the README should instruct to install it.
    nlp = None

# Read product data using a relative path (was '/data/sample30.csv' which fails
# on Windows and when repository root is current working directory)
DATA_CSV = os.path.join(os.path.dirname(__file__), 'data', 'sample30.csv')
try:
    product_df = pd.read_csv(DATA_CSV, sep=',')
except FileNotFoundError:
    # Fallback to a relative path without module dir (in case cwd is project root)
    try:
        product_df = pd.read_csv('data/sample30.csv', sep=',')
    except Exception:
        product_df = pd.DataFrame()

# Text Preprocessing Functions

def remove_special_characters(text, remove_digits=True):
    """Remove special characters (and optionally digits) from text."""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    return re.sub(pattern, '', text)

def to_lowercase(words):
    """Convert list of words to lowercase."""
    return [word.lower() for word in words]

def remove_punctuation_and_splchars(words):
    """Remove punctuation and special characters from list of words."""
    return [remove_special_characters(re.sub(r'[^\w\s]', '', word), True) for word in words if word]

stopword_list = stopwords.words('english')

def remove_stopwords(words):
    """Filter out stopwords from list of words."""
    return [word for word in words if word not in stopword_list]

def stem_words(words):
    """Stem words using Lancaster stemmer."""
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in list of words using WordNet."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    """Normalize words by lowercasing, removing punctuation, and stopwords."""
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    """Lemmatize list of words."""
    return lemmatize_verbs(words)

def _ensure_text_models():
    """Ensure text vectorizers and model are loaded.
    
    Raises:
        ModelNotLoadedError: If any required model artifacts are missing or corrupt
        DataNotFoundError: If pickle files cannot be found
    """
    global count_vector, tfidf_transformer, model
    
    required_files = {
        'CountVectorizer': ('count_vector.pkl', 'count_vector'),
        'TF-IDF Transformer': ('tfidf_transformer.pkl', 'tfidf_transformer'),
        'Sentiment Model': ('model.pkl', 'model')
    }
    
    for model_name, (filename, var_name) in required_files.items():
        if globals()[var_name] is None:
            model_path = os.path.join(os.path.dirname(__file__), 'pickle_files', filename)
            
            if not os.path.exists(model_path):
                raise DataNotFoundError(
                    f"Missing {model_name} file: {filename}\n"
                    f"Please ensure all model files are in the pickle_files directory."
                )
                
            try:
                with open(model_path, 'rb') as f:
                    try:
                        globals()[var_name] = pk.load(f)
                    except Exception as e:
                        raise ModelNotLoadedError(
                            f"Failed to load {model_name} from {filename}: {str(e)}\n"
                            f"The model file may be corrupted."
                        )
            except Exception as e:
                raise ModelNotLoadedError(
                    f"Error accessing {model_name} file: {str(e)}\n"
                    f"Please check file permissions and try again."
                )


def model_predict(text):
    """Predict sentiment label for given text using trained model.

    `text` is expected to be an iterable of strings.
    """
    _ensure_text_models()
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    return model.predict(tfidf_vector)

def normalize_and_lemmaize(input_text):
    """Normalize and lemmatize input text."""
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    return ' '.join(lemmatize(words))

def recommend_products(user_name: str) -> pd.DataFrame:
    """Generate top 20 product recommendations with sentiment predictions for a user.
    
    Args:
        user_name: The username to generate recommendations for
        
    Returns:
        DataFrame containing recommended products with their predicted sentiments
        
    Raises:
        DataNotFoundError: If recommendation matrix file is missing
        ModelNotLoadedError: If recommendation matrix cannot be loaded
        UserNotFoundError: If user_name is not found in the matrix
        ValueError: If user_name is empty or invalid
    """
    if not isinstance(user_name, str) or not user_name.strip():
        raise ValueError("User name must be a non-empty string")
        
    global recommend_matrix
    # Lazy-load the user recommendation matrix
    if recommend_matrix is None:
        r_path = os.path.join(os.path.dirname(__file__), 'pickle_files', 'user_final_rating.pkl')
        if not os.path.exists(r_path):
            raise DataNotFoundError(
                "Missing recommendation matrix file: user_final_rating.pkl\n"
                "Please ensure the file is present in the pickle_files directory."
            )
        try:
            with open(r_path, 'rb') as f:
                try:
                    recommend_matrix = pk.load(f)
                except Exception as e:
                    raise ModelNotLoadedError(
                        f"Failed to load recommendation matrix: {str(e)}\n"
                        "The matrix file may be corrupted."
                    )
        except Exception as e:
            raise ModelNotLoadedError(
                f"Error accessing recommendation matrix file: {str(e)}\n"
                "Please check file permissions and try again."
            )

    if user_name not in recommend_matrix.index:
        raise UserNotFoundError(
            f"User '{user_name}' not found in recommendation matrix.\n"
            "Please check the username and try again with a valid user."
        )

    try:
        # Get top 20 recommendations
        product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[:20])
        
        # Filter product data
        product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
        if product_frame.empty:
            raise ModelError("No product data found for recommendations")
            
        # Prepare output with sentiment predictions
        output_df = product_frame[['name', 'reviews_text']]
        output_df['lemmatized_text'] = output_df['reviews_text'].map(normalize_and_lemmaize)
        output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
        
        return output_df
        
    except Exception as e:
        if isinstance(e, ModelError):
            raise
        raise ModelError(f"Error generating recommendations: {str(e)}")

def top5_products(df: pd.DataFrame) -> pd.DataFrame:
    """Return top 5 products with highest positive sentiment percentage.
    
    Args:
        df: DataFrame containing product recommendations with sentiment predictions
            Must have columns: 'name', 'predicted_sentiment'
            
    Returns:
        DataFrame containing the top 5 products with highest positive sentiment
        
    Raises:
        ValueError: If input DataFrame is empty or missing required columns
        ModelError: If computation fails due to data issues
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be empty")
        
    required_cols = {'name', 'predicted_sentiment'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
    try:
        # Calculate total reviews per product
        total_product = df.groupby(['name']).agg('count')
        
        # Group by product and sentiment
        rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count').reset_index()
        
        # Merge and calculate percentages
        merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
        merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
        
        # Sort and filter top 5 positive sentiment products
        merge_df = merge_df.sort_values(ascending=False, by='%percentage')
        result = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
        
        if result.empty:
            raise ModelError("No products found with positive sentiment")
            
        return result
        
    except Exception as e:
        if isinstance(e, ModelError):
            raise
        raise ModelError(f"Error calculating top products: {str(e)}")
