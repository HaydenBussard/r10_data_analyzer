#!/usr/bin/env python3
"""
Garmin R10 Range Session Analyzer

Processes a directory of Excel and/or CSV exports from Garmin Golf app to calculate weighted
performance metrics for each club.

"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Set your realistic distance ranges for each club (units will be what you use in the Garmin Golf app)
CLUB_RANGES = {
    "Driver": (0, 400),
    "3 Wood": (0, 400),
    "3 Hybrid": (0, 400),
    "3 Iron": (0, 400),
    "4 Iron": (0, 400),
    "5 Iron": (0, 400),
    "6 Iron": (0, 400),
    "7 Iron": (0, 400),
    "8 Iron": (0, 400),
    "9 Iron": (0, 400),
    "Pitching Wedge": (0, 400),
    # Add more clubs as needed
}

def get_trajectory_profile(launch_angle, spin_rate):
    """Determine trajectory profile based on launch angle and spin rate.
    
    Args:
        launch_angle (float): Club's average launch angle
        spin_rate (float): Club's average spin rate
        
    Returns:
        str: Trajectory profile description
    """
    if launch_angle > 15 and spin_rate > 5000:
        return "High Launch, High Spin"
    elif 10 <= launch_angle <= 15 and 3000 <= spin_rate <= 5000:
        return "Mid Launch, Mid Spin"
    else:
        return "Low Launch, Low Spin"

def get_shot_shape(face_to_path, spin_axis):
    """Determine typical shot shape based on face-to-path and spin axis.
    
    Args:
        face_to_path (float): Club's average face-to-path angle
        spin_axis (float): Club's average spin axis
        
    Returns:
        str: Shot shape description (Draw/Fade/Hook/Slice)
    """
    # Determine draw/hook
    if spin_axis < -3:
        if abs(face_to_path) > 3:
            return "Hook"
        else:
            return "Draw"
    # Determine fade/slice
    elif spin_axis > 3:
        if abs(face_to_path) > 3:
            return "Slice"
        else:
            return "Fade"
    # Determine straight
    else:
        return "Straight"

def get_consistency(carry_stddev):
    """Rate consistency based on standard deviation of carry distance.
    
    Args:
        carry_stddev (float): Standard deviation of carry distance
        
    Returns:
        str: Consistency rating (High/Medium/Low)
    """
    if carry_stddev < 5:
        return "High"
    elif 5 <= carry_stddev <= 10:
        return "Medium"
    else:
        return "Low"

def get_risk_level(deviation_lr, deviation_stddev):
    """Determine risk level based on left/right deviation patterns.
    
    Args:
        deviation_lr (float): Average left/right deviation
        deviation_stddev (float): Standard deviation of left/right deviation
        
    Returns:
        str: Risk level (Low/Medium/High)
    """
    abs_deviation_lr = abs(deviation_lr)
    if abs_deviation_lr < 10 and deviation_stddev < 5:
        return "Low Risk"
    elif (10 <= abs_deviation_lr <= 20) or (5 <= deviation_stddev <= 10):
        return "Medium Risk"
    else:
        return "High Risk"

def load_session_data(folder_path):
    """Load all Excel/CSV session files from the specified folder.
    
    Args:
        folder_path (str): Path to folder containing Garmin R10 Excel/CSV exports
        
    Returns:
        pd.DataFrame: Combined data from all session files

    Notes:
        skiprows is used for the double-row headers in the r10 range session files.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.csv'))]
    if not files:
        raise FileNotFoundError(f"No CSV or Excel files found in {folder_path}")
    
    dfs = []
    for f in files:
        full_path = os.path.join(folder_path, f)
        if f.endswith('.xlsx'):
            df = pd.read_excel(full_path, header=0, skiprows=[1])
        else:
            df = pd.read_csv(full_path, header=0, skiprows=[1])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def filter_club_data(df, club_ranges):
    """Filter out unrealistic shots based on club distance ranges.
    
    Args:
        df (pd.DataFrame): Combined session data
        club_ranges (dict): Dictionary of club types with min/max distance ranges
        
    Returns:
        pd.DataFrame: Filtered data with only realistic shots
    """
    filtered_data = pd.DataFrame()
    for club, (min_dist, max_dist) in club_ranges.items():
        club_data = df[(df["Club Type"] == club) & 
                      (df["Carry Distance"] >= min_dist) & 
                      (df["Carry Distance"] <= max_dist)]
        filtered_data = pd.concat([filtered_data, club_data], ignore_index=True)
    return filtered_data

def calculate_weights(df, decay_factor):
    """Calculate weights by recency.

    Args:
        df (pd.DataFrame): Data to analyze
        decay_factor: DECAY_FACTOR variable for weight distribution

    Returns:
        df (pd.DataFrame): same data with assigned weight values for shots
    """
    df = df.sort_values(by='Date').reset_index(drop=True)
    df['recency_rank'] = range(len(df)-1, -1, -1)
    df['weight'] = np.exp(-decay_factor * df['recency_rank'])
    return df

def weighted_mean(df, value_col, weight_col):
    """Calculate weighted mean.
    
    Args:
        df (pd.DataFrame): Data to analyze
        value_col (str): Column name for values to average
        weight_col (str): Column name for weights
        
    Returns:
        float: Weighted mean of the values
    """
    return np.average(df[value_col], weights=df[weight_col])

def weighted_std(df, value_col, weight_col):
    """Calculate weighted standard deviation.
    
    Args:
        df (pd.DataFrame): Data to analyze
        value_col (str): Column name for values
        weight_col (str): Column name for weights
        
    Returns:
        float: Weighted standard deviation
    """
    avg = weighted_mean(df, value_col, weight_col)
    variance = np.average((df[value_col] - avg) ** 2, weights=df[weight_col])
    return np.sqrt(variance)

def analyze_club_performance(filtered_data, decay_factor, num_shots=100):
    """
    Calculate performance metrics for each club, using a weighted average of the
    most recent 'num_shots' for each club.
    
    Args:
        filtered_data (pd.DataFrame): Filtered session data
        decay_factor (float): The decay factor for the weighting function
        num_shots (int): The number of recent shots to use for each club's analysis
        
    Returns:
        pd.DataFrame: Club-by-club performance metrics
    """
    club_stats = []
    club_dist_only = []
    
    # Group the filtered data by club
    grouped_clubs = filtered_data.groupby("Club Type")
    
    for club, group_df in grouped_clubs:
        # Sort the shots for the current club by date and take the most recent 'num_shots'
        recent_shots = group_df.sort_values(by='Date', ascending=False).head(num_shots).copy()
        
        # Ensure there are enough shots to analyze
        if len(recent_shots) == 0:
            continue
            
        # Apply the weighting to this subset of data
        weighted_df = calculate_weights(recent_shots, decay_factor)
        
        # Calculate the weighted average and standard deviation
        avg_carry = np.average(weighted_df["Carry Distance"], weights=weighted_df["weight"])
        carry_stddev = np.sqrt(np.average((weighted_df["Carry Distance"] - avg_carry)**2, weights=weighted_df["weight"]))
        
        avg_total = np.average(weighted_df["Total Distance"], weights=weighted_df["weight"])
        avg_deviation_lr = np.average(weighted_df["Carry Deviation Distance"], weights=weighted_df["weight"])
        deviation_stddev = np.sqrt(np.average((weighted_df["Carry Deviation Distance"] - avg_deviation_lr)**2, weights=weighted_df["weight"]))
        
        avg_launch_angle = np.average(weighted_df["Launch Angle"], weights=weighted_df["weight"])
        avg_spin_rate = np.average(weighted_df["Spin Rate"], weights=weighted_df["weight"])
        
        avg_face_to_path = np.average(weighted_df["Face to Path"], weights=weighted_df["weight"])
        avg_spin_axis = np.average(weighted_df["Spin Axis"], weights=weighted_df["weight"])

        # Find the maximum values from the `recent_shots` DataFrame
        max_carry = recent_shots["Carry Distance"].max()
        max_total = recent_shots["Total Distance"].max()
        
        # Create a dictionary of the calculated stats
        club_stats.append({
            "Club": club,
            "Avg Carry": avg_carry,
            "Avg Total": avg_total,
            "Max Carry": max_carry,
            "Max Total": max_total,
            "Deviation (L/R)": avg_deviation_lr,
            "Carry Stddev": carry_stddev,
            "Trajectory Profile": get_trajectory_profile(avg_launch_angle, avg_spin_rate),
            "Shot Shape": get_shot_shape(avg_face_to_path, avg_spin_axis),
            "Consistency": get_consistency(carry_stddev),
            "Risk Level": get_risk_level(avg_deviation_lr, deviation_stddev),
            "Count": len(recent_shots)
        })

        # Create a dictionary of avg and max distances
        club_dist_only.append({
            "Club": club,
            "Avg Carry": avg_carry,
            "Avg Total": avg_total,
            "Max Carry": max_carry,
            "Max Total": max_total
        })

    # Return both dictionaries as dataframes
    return pd.DataFrame(club_stats), pd.DataFrame(club_dist_only)

def main():
    """Main execution function."""
    print("Loading and processing Garmin R10 session data...")
    
    try:
        # Load and prep data
        all_data = load_session_data(FOLDER_PATH)
        all_data['Date'] = pd.to_datetime(all_data['Date'])
        filtered_data = filter_club_data(all_data, CLUB_RANGES)
        
        # Analyze performance and round data to whole numbers
        club_stats, club_dist_only = analyze_club_performance(filtered_data, DECAY_FACTOR, NUM_CLUB_SHOTS)
        club_stats = club_stats.round(0)
        club_dist_only = club_dist_only.round(0)

        # Save both dataframes to Excel
        with pd.ExcelWriter(OUTPUT_FILE) as writer:
            # First sheet: Distances only
            club_dist_only.to_excel(writer, sheet_name='Distances Only', index=False)
            
            # Second sheet: All stats
            club_stats.to_excel(writer, sheet_name='All Stats', index=False)
        
        print(f"Club yardages saved to {OUTPUT_FILE} with two tabs.")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    # Configuration - modify these paths
    FOLDER_PATH = "/Users/hayde/OneDrive/Golf Practice/Driving Range"
    OUTPUT_FILE = "/Users/hayde/OneDrive/Golf Practice/club_yardages.xlsx"

    NUM_CLUB_SHOTS = 100
    DECAY_FACTOR = 0.05

    main()
