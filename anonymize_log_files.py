# coding: utf-8
import pandas as pd
import os
import re
from typing import Optional


def anonymize_log_file(input_filepath: str, output_filepath: str) -> Optional[str]:
    """
    Reads a text-based log file, finds all timestamps matching 'YYYY-MM-DD HH:MM:SS.microseconds', 
    anonymizes them by subtracting the first timestamp found, and saves the modified content.

    Args:
        input_filepath: Path to the original log file.
        output_filepath: Path where the new, anonymized log file will be saved.

    Returns:
        The output filepath on success, None on failure.
    """
    print(f"  > Processing LOG file: {os.path.basename(input_filepath)}")
    
    # Regex pattern for the timestamp format (YYYY-MM-DD HH:MM:SS.microseconds)
    TIMESTAMP_PATTERN = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})'
    
    try:
        with open(input_filepath, 'r') as f:
            content = f.read()

        # 1. Find all timestamps in the entire file
        all_timestamp_strings = re.findall(TIMESTAMP_PATTERN, content)
        
        if not all_timestamp_strings:
            print("  > Warning: No timestamps found in log file. Copying file as-is.")
            # Simply copy the file if no timestamps are found
            with open(output_filepath, 'w') as f_out:
                f_out.write(content)
            return output_filepath

        # 2. Find the reference point (the very first valid timestamp)
        unique_timestamp_strings = sorted(list(set(all_timestamp_strings)))
        
        # Convert unique strings to datetime objects
        unique_datetimes = pd.to_datetime(unique_timestamp_strings, errors='coerce', infer_datetime_format=True)
        
        # Filter out NaT values and get the earliest valid time
        valid_datetimes = unique_datetimes[pd.notna(unique_datetimes)]
        
        if valid_datetimes.empty:
            print("  > Error: Could not parse any timestamp in the log file.")
            return None

        first_timestamp = valid_datetimes.min()
        
        # 3. Create a map from absolute time string to relative time string
        anonymization_map = {}
        for abs_str, abs_dt in zip(unique_timestamp_strings, unique_datetimes):
            if pd.isna(abs_dt):
                # If parsing failed for a specific string, don't change it
                anonymization_map[abs_str] = abs_str
            else:
                relative_td = abs_dt - first_timestamp
                anonymization_map[abs_str] = format_timedelta_as_zeroed_timestamp(relative_td)

        # 4. Perform the substitution using the pre-calculated map
        def replacer(match):
            return anonymization_map.get(match.group(0), match.group(0))

        anonymized_content = re.sub(TIMESTAMP_PATTERN, replacer, content)

        # 5. Save the anonymized content
        with open(output_filepath, 'w') as f_out:
            f_out.write(anonymized_content)

        return output_filepath

    except FileNotFoundError:
        print(f"Error: Input log file not found at {input_filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {input_filepath}: {e}")
        return None


def process_data_directory(source_dir: str):
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

    print(f"--- Starting Directory Processing ---")
    print(f"Source Directory: {source_dir}")

    # Walk through the source directory structure
    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)

        for file in files:
            if file.lower().endswith('.log'):
                # Call the new anonymization function for LOG
                input_filepath = os.path.join(root, file)
                output_filepath = os.path.join(root, file.lower().replace('.log', '_anonymized.log'))
                anonymize_log_file(input_filepath, output_filepath)
    
    print(f"\n--- Directory Processing Complete. ---")


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
    
process_data_directory(SOURCE_DIR)