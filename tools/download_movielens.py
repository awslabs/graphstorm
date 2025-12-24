"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    MovieLens Dataset Download and Conversion Tool

    This tool downloads the MovieLens-100k dataset and converts all data files
    (users, movies, ratings) to parquet format for the
    and gen_mitra_embedding.py.
    
    Output Structure:
    -----------------
    output_dir/
    ├── user/
    │   └── users.parquet
    ├── movie/
    │   └── movies.parquet
    └── rating/
        └── ratings.parquet
"""

import os
import argparse
import urllib.request
import zipfile
import pandas as pd


def download_movielens_100k(raw_dir):
    """
    Download and extract MovieLens 100k dataset if not already present.
    
    Parameters
    ----------
    raw_dir : str
        Directory where the dataset should be stored
        
    Returns
    -------
    str
        Path to the directory containing the extracted ml-100k folder
    """
    ml_dir = os.path.join(raw_dir, 'ml-100k')
    
    # Check if already downloaded
    if os.path.exists(os.path.join(ml_dir, 'u.user')):
        print(f"MovieLens 100k dataset already exists at {ml_dir}")
        return raw_dir
    
    # Create directory if it doesn't exist
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download URL
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(raw_dir, 'ml-100k.zip')
    
    print(f"Downloading MovieLens 100k dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
        
        # Extract zip file
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        # Remove zip file
        os.remove(zip_path)
        print(f"MovieLens 100k dataset extracted to {ml_dir}")
        
        return raw_dir
    except Exception as e:
        raise RuntimeError(f"Failed to download MovieLens 100k dataset: {e}")


def convert_movielens_to_parquet(raw_data_path, output_dir):
    """
    Convert all MovieLens raw data files to parquet format.
    
    This function converts users, movies, and ratings data into separate
    parquet files organized by node/edge type for GraphStorm gconstruct.
    
    Parameters
    ----------
    raw_data_path : str
        Path to the directory containing MovieLens files (e.g., data/ml-100k/ml-100k)
    output_dir : str
        Directory where parquet files will be saved in subdirectories by type
        
    Returns
    -------
    dict
        Dictionary with paths to converted files for each type
    """
    os.makedirs(output_dir, exist_ok=True)
    converted_files = {}
    
    # Convert user data
    print("\n" + "="*70)
    print("Converting User Data")
    print("="*70)
    
    user_file = os.path.join(raw_data_path, 'u.user')
    if not os.path.exists(user_file):
        raise FileNotFoundError(f"u.user file not found at {user_file}")
    
    # u.user format: user_id | age | gender | occupation | zip_code
    df = pd.read_csv(user_file, sep='|', header=None, 
                     names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    
    print(f"Loaded {len(df)} users")
    
    # Convert gender to numeric (M=1, F=0)
    df['gender_numeric'] = (df['gender'] == 'M').astype(int)
    
    # Load occupation mapping
    occupation_file = os.path.join(raw_data_path, 'u.occupation')
    if os.path.exists(occupation_file):
        with open(occupation_file, 'r') as f:
            occupations = [line.strip() for line in f.readlines()]
        occupation_map = {occ: idx for idx, occ in enumerate(occupations)}
        df['occupation_numeric'] = df['occupation'].map(occupation_map)
    else:
        # Fallback: create mapping from unique values
        unique_occupations = df['occupation'].unique()
        occupation_map = {occ: idx for idx, occ in enumerate(unique_occupations)}
        df['occupation_numeric'] = df['occupation'].map(occupation_map)
    
    # Bin age into groups for classification
    age_bins = [0, 18, 25, 35, 45, 55, 100]
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=False)
    
    # Create output directory for users
    user_output_dir = os.path.join(output_dir, 'user')
    os.makedirs(user_output_dir, exist_ok=True)
    
    # Save to parquet
    user_output_path = os.path.join(user_output_dir, 'users.parquet')
    df.to_parquet(user_output_path, index=False)
    converted_files['user'] = user_output_path
    
    print(f"Saved user data to: {user_output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    
    # Convert movie data
    print("\n" + "="*70)
    print("Converting Movie Data")
    print("="*70)
    
    item_file = os.path.join(raw_data_path, 'u.item')
    if not os.path.exists(item_file):
        raise FileNotFoundError(f"u.item file not found at {item_file}")
    
    # u.item format: movie_id | title | release_date | video_release_date | IMDb_URL | 19 genre columns
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    col_names = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
    
    df = pd.read_csv(item_file, sep='|', header=None, names=col_names, encoding='latin-1')
    
    print(f"Loaded {len(df)} movies")
    
    # Extract year from release_date
    df['release_year'] = pd.to_datetime(df['release_date'], format='%d-%b-%Y', errors='coerce').dt.year
    df['release_year'] = df['release_year'].fillna(0).astype(int)
    
    # Create genre count and primary genre
    df['genre_count'] = df[genre_cols].sum(axis=1)
    
    # Get primary genre (first genre that is 1)
    def get_primary_genre(row):
        for idx, genre in enumerate(genre_cols):
            if row[genre] == 1:
                return idx
        return -1  # No genre
    
    df['primary_genre'] = df.apply(get_primary_genre, axis=1)
    
    # Create output directory for movies
    movie_output_dir = os.path.join(output_dir, 'movie')
    os.makedirs(movie_output_dir, exist_ok=True)
    
    # Save to parquet
    movie_output_path = os.path.join(movie_output_dir, 'movies.parquet')
    df.to_parquet(movie_output_path, index=False)
    converted_files['movie'] = movie_output_path
    
    print(f"Saved movie data to: {movie_output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    
    # Convert ratings data
    print("\n" + "="*70)
    print("Converting Ratings Data")
    print("="*70)
    
    rating_file = os.path.join(raw_data_path, 'u.data')
    if not os.path.exists(rating_file):
        raise FileNotFoundError(f"u.data file not found at {rating_file}")
    
    # u.data format: user_id | movie_id | rating | timestamp
    df = pd.read_csv(rating_file, sep='\t', header=None, 
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    print(f"Loaded {len(df)} ratings")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Create output directory for ratings
    rating_output_dir = os.path.join(output_dir, 'rating')
    os.makedirs(rating_output_dir, exist_ok=True)
    
    # Save to parquet
    rating_output_path = os.path.join(rating_output_dir, 'ratings.parquet')
    df.to_parquet(rating_output_path, index=False)
    converted_files['rating'] = rating_output_path
    
    print(f"Saved rating data to: {rating_output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    
    return converted_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and convert MovieLens-100k dataset to parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Download and convert all MovieLens data
            python tools/download_movielens.py --output-dir data/ml-100k-parquet
            
            # Use existing downloaded data
            python tools/download_movielens.py --raw-data-path data/ml-100k/ml-100k \\
                --output-dir data/ml-100k-parquet --skip-download

        Output Structure:
            output-dir/
            ├── user/
            │   └── users.parquet
            ├── movie/
            │   └── movies.parquet
            └── rating/
                └── ratings.parquet

        Usage with gen_mitra_embedding.py:
            # Generate embeddings for users
            python tools/gen_mitra_embedding.py \\
                --dataset_path data/ml-100k-parquet \\
                --target-ntype user \\
                --label-name gender \\
                --node-id-col user_id
            """
    )
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory where parquet files will be saved (creates user/, movie/, rating/ subdirectories)")
    parser.add_argument("--raw-data-path", type=str, default=None,
                       help="Path to existing MovieLens raw data directory (containing u.user, u.item, u.data files). If not specified, will download to output-dir/raw/")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download step and use existing raw data at raw-data-path")
    
    args = parser.parse_args()
    print("="*70)
    print("MovieLens-100k Dataset Converter")
    print("="*70)
    
    # Determine raw data path
    if args.raw_data_path:
        raw_path = args.raw_data_path
        if not args.skip_download:
            print(f"Warning: raw-data-path specified but skip-download not set. Will attempt download if data missing.")
            download_movielens_100k(os.path.dirname(raw_path))
    else:
        # Download to a subdirectory of output-dir
        download_dir = os.path.join(args.output_dir, 'raw')
        download_movielens_100k(download_dir)
        raw_path = os.path.join(download_dir, 'ml-100k')
    
    # Convert all data to parquet
    converted_files = convert_movielens_to_parquet(raw_path, args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: MovieLens data converted to parquet format")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nConverted files:")
    for data_type, file_path in converted_files.items():
        print(f"  {data_type}: {file_path}")
    print(f"\n{'='*70}")
    print(f"Next Steps:")
    print(f"{'='*70}")
    print(f"Generate Mitra embeddings for users:")
    print(f"  python tools/gen_mitra_embedding.py \\")
    print(f"      --dataset_path {args.output_dir} \\")
    print(f"      --target-ntype user \\")
    print(f"      --label-name gender \\")
    print(f"      --node-id-col user_id")
    print(f"{'='*70}")
