import pandas as pd
import numpy as np
import re


class HP330:
    """Class for reading and processing HP330 spectral data files."""
    
    def __init__(self):
        """Initialize the HP330 class."""
        self.spectral_data = None
        self.df = None
    
    def read_file(self, file_path):
        """
        Read a CSV file and extract spectral data into a DataFrame.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing spectral data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns 'wavelength' and 'intensity', sorted by wavelength
        """
        # Read all lines
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Extract spectral data - lines matching pattern: number(mW/m2/nm),number
        spectral_data = []
        wavelength_pattern = re.compile(r'(\d+)\(mW/m2/nm\),([\d.]+)')
        
        for line in lines:
            match = wavelength_pattern.search(line)
            if match:
                wavelength = int(match.group(1))
                intensity = float(match.group(2))
                spectral_data.append({'wavelength': wavelength, 'intensity': intensity})
        
        # Create DataFrame
        df = pd.DataFrame(spectral_data)
        
        # Sort by wavelength to ensure proper order
        df = df.sort_values('wavelength').reset_index(drop=True)
        
        # Store as instance variables
        self.spectral_data = spectral_data
        self.df = df
        
        return df


class Filter:
    """Class for calculating filter attenuation from spectral data files."""
    
    def __init__(self):
        """Initialize the Filter class."""
        self.attenuation_df = None
    
    def calculate_attenuation(self, transmitted_file, incident_file):
        """
        Read two spectral files and calculate attenuation values.
        
        Attenuation is calculated as: transmitted_intensity / incident_intensity
        This gives transmittance (0-1), where lower values indicate more attenuation.
        
        Parameters:
        -----------
        transmitted_file : str
            Path to the CSV file containing transmitted/filtered spectral data
        incident_file : str
            Path to the CSV file containing incident/original spectral data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns 'wavelength' and 'attenuation', sorted by wavelength
        """
        # Read both files using HP330 class
        hp330 = HP330()
        transmitted_df = hp330.read_file(transmitted_file)
        incident_df = hp330.read_file(incident_file)
        
        # Merge dataframes on wavelength to ensure matching wavelengths
        merged_df = pd.merge(
            transmitted_df[['wavelength', 'intensity']],
            incident_df[['wavelength', 'intensity']],
            on='wavelength',
            suffixes=('_transmitted', '_incident')
        )
        
        # Calculate attenuation (transmittance ratio)
        # Avoid division by zero
        merged_df['attenuation'] = merged_df['intensity_transmitted'] / merged_df['intensity_incident'].replace(0, np.nan)
        
        # Create result dataframe with wavelength and attenuation
        attenuation_df = merged_df[['wavelength', 'attenuation']].copy()
        
        # Sort by wavelength
        attenuation_df = attenuation_df.sort_values('wavelength').reset_index(drop=True)
        
        # Store as instance variable
        self.attenuation_df = attenuation_df
        
        return attenuation_df

