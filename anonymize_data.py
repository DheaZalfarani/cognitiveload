# coding: utf-8
import pandas as pd
import os
from typing import Optional


def anonymize_timestamps(input_filepath: str, output_filepath: str, timestamp_column_index: int = 0) -> Optional[str]:
    """
    Loads a headerless CSV file, converts the absolute timestamps in the specified
    column index to relative time (timedelta) by subtracting the first timestamp 
    from all subsequent entries, and saves the result to a new, headerless CSV file 
    with the relative time formatted as '0000-00-00 HH:MM:SS.MILLISECONDS'.

    This process effectively anonymizes the data by removing all real-world time 
    references while preserving the temporal relationships between data points.

    Args:
        input_filepath: Path to the original CSV file (must be headerless).
        output_filepath: Path where the new, anonymized CSV file will be saved.
        timestamp_column_index: The zero-based index of the column containing the timestamps. 
                                Defaults to 0 (the first column).

    Returns:
        The output filepath on success, None on failure.
    """
    print(f"  > Processing file: {os.path.basename(input_filepath)}")
    
    # --- Robust Error Handling ---
    try:
        # 1. Load the CSV file without expecting a header
        df = pd.read_csv(input_filepath, header=None)
        
        # The column name will be the integer index passed (e.g., 0)
        timestamp_column = timestamp_column_index

        # 2. Validate the timestamp column exists (check if the index is out of bounds)
        if timestamp_column not in df.columns:
            print(f"Error: Column index {timestamp_column} is out of bounds for the CSV file.")
            return None

        # 3. Convert the specified column to datetime objects
        df['Absolute_Time'] = pd.to_datetime(df[timestamp_column], errors='coerce', infer_datetime_format=True)
        
        # 4. Find the reference point (the very first valid timestamp)
        first_timestamp = df['Absolute_Time'].min()

        if pd.isna(first_timestamp):
            print("Error: No valid timestamps found to establish a zero-point.")
            return None

        # 5. Calculate the time difference (timedelta) for every entry
        relative_time = df['Absolute_Time'] - first_timestamp
        
        # 6. Overwrite the original timestamp column (index 0) with the formatted relative time string
        df[timestamp_column] = relative_time.apply(format_timedelta_as_zeroed_timestamp)

        # 7. Clean up the DataFrame for export: Drop the temporary 'Absolute_Time' column and save
        df = df.drop(columns=['Absolute_Time'])
        df.to_csv(output_filepath, index=False, header=False)

        # print(f"Success: Saved to {output_filepath}") # Suppress success message for directory processor
        return output_filepath

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {input_filepath}: {e}")
        return None


def process_data_directory(source_dir: str, output_folder_name: str = 'anonymized_data'):
    """
    Traverses a source directory recursively, mirrors the folder structure 
    in a new 'anonymized_data' folder (created at the same level as source_dir), 
    and anonymizes all CSV files found using the anonymize_timestamps function.

    Args:
        source_dir: The root directory containing the raw data.
        output_folder_name: The name for the new output directory.
    """
    source_dir = os.path.abspath(source_dir)
    parent_dir = os.path.dirname(source_dir)
    output_root_dir = os.path.join(parent_dir, output_folder_name)

    print(f"--- Starting Directory Processing ---")
    print(f"Source Directory: {source_dir}")
    print(f"Output Directory: {output_root_dir}\n")

    # Create the root output directory if it doesn't exist
    os.makedirs(output_root_dir, exist_ok=True)

    # Walk through the source directory structure
    for root, dirs, files in os.walk(source_dir):
        # 1. Determine the relative path from the source root
        relative_path = os.path.relpath(root, source_dir)
        
        # 2. Construct the corresponding output directory path
        output_dir = os.path.join(output_root_dir, relative_path)

        # 3. Create the mirrored directory structure in the output location
        # os.walk already ensures the parent directories (relative_path) exist, 
        # but we use makedirs here for safety, though often not strictly necessary 
        # inside os.walk loop, it guarantees all subfolders exist.
        os.makedirs(output_dir, exist_ok=True)

        # 4. Process all CSV files in the current folder
        for file in files:
            if file.lower().endswith('.csv'):
                input_filepath = os.path.join(root, file)
                
                # Construct the new output filename and path
                base_name, ext = os.path.splitext(file)
                output_filename = base_name + '_anonymized' + ext
                output_filepath = os.path.join(output_dir, output_filename)
                
                # Call the anonymization function
                anonymize_timestamps(input_filepath, output_filepath)
    
    print(f"\n--- Directory Processing Complete. Results saved in: {output_root_dir} ---")


def format_timedelta_as_zeroed_timestamp(td: pd.Timedelta) -> str:
    """
    Helper function to convert a Timedelta (duration) object into the desired 
    string format: '0000-00-00 HH:MM:SS.microseconds', maintaining high precision 
    by calculating components directly from the total nanosecond value.
    """
    if pd.isna(td):
        return '' 

    total_ns = td.value
    
    # Relative time should not be negative
    if total_ns < 0:
        return '0000-00-00 00:00:00.000000'

    # 1. Separate the total seconds from the fractional nanoseconds
    total_seconds = total_ns // 1_000_000_000 # Integer seconds
    frac_ns_remainder = total_ns % 1_000_000_000 # Nanoseconds remainder

    # 2. Calculate the microseconds (6 digits) using explicit rounding (add 500 ns and integer divide by 1000)
    # This prevents non-zero times from rounding down to zero and is more robust than floating-point math.
    microseconds = (frac_ns_remainder + 500) // 1000

    # 3. Handle potential carry-over if rounding results in 1,000,000 (i.e., a full second)
    if microseconds >= 1_000_000:
        total_seconds += 1
        microseconds -= 1_000_000

    # 4. Convert total seconds into HH:MM:SS (handles durations > 24 hours correctly)
    total_minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    total_hours = total_minutes // 60
    minutes = total_minutes % 60
    hours = total_hours # Full hour count (can exceed 99)

    # Final string composition (HH is minimum 2 digits, but will expand if needed)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"
    
    return f"0000-00-00 {time_str}"


# Define the source directory name for the example
SOURCE_DIR = 'DIR_TO_ANONYMIZE_FROM'
ANONYMIZED_DIR = 'DIR_TO_ANONYMIZE_TO'
    
process_data_directory(SOURCE_DIR, ANONYMIZED_DIR)