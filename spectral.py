import pandas as pd
import numpy as np
import plotly.graph_objects as go
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


class Analysis:
    """Class for analyzing and visualizing spectral data."""
    
    def __init__(self):
        """Initialize the Analysis class."""
        pass
    
    def plot_spectral(self, spectral_df, figsize=(12, 6), color='blue'):
        """
        Plot the spectral data (wavelength vs intensity).
        
        Parameters:
        -----------
        spectral_df : pd.DataFrame
            DataFrame with columns 'wavelength' and 'intensity'
        figsize : tuple, optional
            Figure size (width, height) in pixels. Default is (12, 6) - converted to pixels.
        color : str, optional
            Color of the plot line. Default is 'blue'.
            
        Returns:
        --------
        None
            Displays the interactive plot
        """
        if spectral_df is None or len(spectral_df) == 0:
            raise ValueError("No spectral data provided.")
        
        if 'wavelength' not in spectral_df.columns or 'intensity' not in spectral_df.columns:
            raise ValueError("DataFrame must contain 'wavelength' and 'intensity' columns.")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=spectral_df['wavelength'],
            y=spectral_df['intensity'],
            mode='lines',
            line=dict(color=color, width=2),
            name='Spectral Data'
        ))
        
        fig.update_layout(
            title=dict(text='Spectral Distribution', font=dict(size=14, family='Arial Black')),
            xaxis_title='Wavelength (nm)',
            yaxis_title='Intensity (mW/m²/nm)',
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(range=[spectral_df['wavelength'].min(), spectral_df['wavelength'].max()])
        
        fig.show()
    
    def plot_attenuation(self, attenuation_df, figsize=(12, 6), color='red', ylim=(0, 1), log_scale=True):
        """
        Plot the attenuation data.
        
        Parameters:
        -----------
        attenuation_df : pd.DataFrame
            DataFrame with columns 'wavelength' and 'attenuation'
        figsize : tuple, optional
            Figure size (width, height) in pixels. Default is (12, 6) - converted to pixels.
        color : str, optional
            Color of the plot line. Default is 'red'.
        ylim : tuple, optional
            Y-axis limits (min, max). Default is (0, 1).
        log_scale : bool, optional
            Whether to use logarithmic y-axis scale. Default is True.
            
        Returns:
        --------
        None
            Displays the interactive plot
        """
        if attenuation_df is None or len(attenuation_df) == 0:
            raise ValueError("No attenuation data provided.")
        
        if 'wavelength' not in attenuation_df.columns or 'attenuation' not in attenuation_df.columns:
            raise ValueError("DataFrame must contain 'wavelength' and 'attenuation' columns.")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=attenuation_df['wavelength'],
            y=attenuation_df['attenuation'],
            mode='lines',
            line=dict(color=color, width=2),
            name='Attenuation'
        ))
        
        fig.update_layout(
            title=dict(text='Filter Attenuation', font=dict(size=14, family='Arial Black')),
            xaxis_title='Wavelength (nm)',
            yaxis_title='Attenuation (Transmittance)',
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(range=[attenuation_df['wavelength'].min(), attenuation_df['wavelength'].max()])
        
        yaxis_config = {}
        if log_scale:
            yaxis_config['type'] = 'log'
            if ylim[0] > 0:  # Only set ylim if min is positive (required for log scale)
                yaxis_config['range'] = ylim
        else:
            yaxis_config['range'] = ylim
        fig.update_yaxes(**yaxis_config)
        
        fig.show()
    
    def plot_multiple_attenuation(self, attenuation_dfs, labels=None, figsize=(12, 6), colors=None, ylim=(0, 1), log_scale=True):
        """
        Plot multiple attenuation curves on the same plot.
        
        Parameters:
        -----------
        attenuation_dfs : list of pd.DataFrame
            List of DataFrames, each with columns 'wavelength' and 'attenuation'
        labels : list of str, optional
            Labels for each curve. If None, uses 'Filter 1', 'Filter 2', etc.
        figsize : tuple, optional
            Figure size (width, height) in pixels. Default is (12, 6) - converted to pixels.
        colors : list of str, optional
            Colors for each curve. If None, uses default plotly colors.
        ylim : tuple, optional
            Y-axis limits (min, max). Default is (0, 1).
        log_scale : bool, optional
            Whether to use logarithmic y-axis scale. Default is True.
            
        Returns:
        --------
        None
            Displays the interactive plot
        """
        if not attenuation_dfs or len(attenuation_dfs) == 0:
            raise ValueError("No attenuation data provided.")
        
        # Default colors if not provided
        if colors is None:
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Default labels if not provided
        if labels is None:
            labels = [f'Filter {i+1}' for i in range(len(attenuation_dfs))]
        
        fig = go.Figure()
        
        # Find overall wavelength range
        min_wavelength = min(df['wavelength'].min() for df in attenuation_dfs)
        max_wavelength = max(df['wavelength'].max() for df in attenuation_dfs)
        
        # Plot each curve
        for i, df in enumerate(attenuation_dfs):
            if df is None or len(df) == 0:
                continue
            if 'wavelength' not in df.columns or 'attenuation' not in df.columns:
                raise ValueError(f"DataFrame {i} must contain 'wavelength' and 'attenuation' columns.")
            
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f'Filter {i+1}'
            
            fig.add_trace(go.Scatter(
                x=df['wavelength'],
                y=df['attenuation'],
                mode='lines',
                line=dict(color=color, width=2),
                name=label
            ))
        
        fig.update_layout(
            title=dict(text='Filter Attenuation Comparison', font=dict(size=14, family='Arial Black')),
            xaxis_title='Wavelength (nm)',
            yaxis_title='Attenuation (Transmittance)',
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        fig.update_xaxes(range=[min_wavelength, max_wavelength])
        
        yaxis_config = {}
        if log_scale:
            yaxis_config['type'] = 'log'
            if ylim[0] > 0:  # Only set ylim if min is positive (required for log scale)
                yaxis_config['range'] = ylim
        else:
            yaxis_config['range'] = ylim
        fig.update_yaxes(**yaxis_config)
        
        fig.show()
    
    def remove_filter(self, spectral_df, filter_df):
        """
        Remove a filter from the spectral distribution by dividing intensity by filter attenuation values.
        
        Parameters:
        -----------
        spectral_df : pd.DataFrame
            DataFrame with columns 'wavelength' and 'intensity'
        filter_df : pd.DataFrame
            DataFrame with columns 'wavelength' and 'attenuation' representing filter values.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns 'wavelength' and 'intensity', where intensity has been
            divided by the filter attenuation values. Wavelengths not present in the filter
            will have their intensity divided by 1.0 (no change).
        """
        if spectral_df is None or len(spectral_df) == 0:
            raise ValueError("No spectral data provided.")
        
        if 'wavelength' not in spectral_df.columns or 'intensity' not in spectral_df.columns:
            raise ValueError("Spectral DataFrame must contain 'wavelength' and 'intensity' columns.")
        
        if filter_df is None or len(filter_df) == 0:
            raise ValueError("No filter data provided.")
        
        if 'wavelength' not in filter_df.columns or 'attenuation' not in filter_df.columns:
            raise ValueError("Filter DataFrame must contain 'wavelength' and 'attenuation' columns.")
        
        # Merge spectral and filter dataframes on wavelength
        merged_df = pd.merge(
            spectral_df[['wavelength', 'intensity']],
            filter_df[['wavelength', 'attenuation']],
            on='wavelength',
            how='left'
        )
        
        # Fill NaN values (wavelengths not in filter) with 1.0 (no attenuation)
        merged_df['attenuation'] = merged_df['attenuation'].fillna(1.0)
        
        # Replace very small attenuation values to avoid division by zero
        # Use a small epsilon value (1e-10) to prevent numerical issues
        merged_df['attenuation'] = merged_df['attenuation'].replace(0, np.nan)
        merged_df['attenuation'] = merged_df['attenuation'].fillna(1e-10)
        
        # Apply filter: divide intensity by attenuation values
        merged_df['intensity'] = merged_df['intensity'] / merged_df['attenuation']
        
        # Return filtered spectral dataframe
        result_df = merged_df[['wavelength', 'intensity']].copy()
        
        return result_df
    
    def integrate_spectral(self, spectral_df, wavelength_start, wavelength_end, filter_df=None):
        """
        Integrate the spectral distribution over a given wavelength range.
        Optionally apply a filter by dividing the spectral distribution by filter values.
        
        Parameters:
        -----------
        spectral_df : pd.DataFrame
            DataFrame with columns 'wavelength' and 'intensity'
        wavelength_start : float
            Starting wavelength for integration (nm)
        wavelength_end : float
            Ending wavelength for integration (nm)
        filter_df : pd.DataFrame, optional
            DataFrame with columns 'wavelength' and 'attenuation' representing filter values.
            If provided, the spectral intensity will be divided by attenuation values before integration.
            Default is None (no filter applied).
            
        Returns:
        --------
        float
            Integrated spectral intensity over the specified wavelength range.
            Units: mW/m² (if no filter) or filtered mW/m² (if filter applied)
        """
        if spectral_df is None or len(spectral_df) == 0:
            raise ValueError("No spectral data provided.")
        
        if 'wavelength' not in spectral_df.columns or 'intensity' not in spectral_df.columns:
            raise ValueError("Spectral DataFrame must contain 'wavelength' and 'intensity' columns.")
        
        if wavelength_start >= wavelength_end:
            raise ValueError("wavelength_start must be less than wavelength_end.")
        
        # Apply filter if provided
        if filter_df is not None:
            spectral_df = self.remove_filter(spectral_df, filter_df)
        
        # Filter spectral data to the specified wavelength range
        mask = (spectral_df['wavelength'] >= wavelength_start) & (spectral_df['wavelength'] <= wavelength_end)
        filtered_spectral = spectral_df[mask].copy()
        
        if len(filtered_spectral) == 0:
            raise ValueError(f"No data points found in wavelength range [{wavelength_start}, {wavelength_end}] nm.")
        
        # Extract values for integration
        intensity_values = filtered_spectral['intensity'].values
        wavelength_values = filtered_spectral['wavelength'].values
        
        # Integrate using trapezoidal rule: ∫ f(x) dx ≈ Σ (f(x_i) + f(x_{i+1})) / 2 * (x_{i+1} - x_i)
        integral = np.trapz(intensity_values, wavelength_values)
        
        return integral
    

