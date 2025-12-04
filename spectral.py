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
    
    def calculate_power_density(self, spectral_df, wavelength_start, wavelength_end, filter_df=None, units='mW/cm2'):
        """
        Calculate power density over a given wavelength range using integration.
        
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
        units : str, optional
            Output units. Options: 'mW/cm2' (default), 'mW/m2', 'W/cm2', 'W/m2'
            
        Returns:
        --------
        float
            Power density over the specified wavelength range in the requested units.
        """
        # Integrate spectral data
        integral_m2 = self.integrate_spectral(spectral_df, wavelength_start, wavelength_end, filter_df)
        
        # Convert to requested units
        if units == 'mW/cm2':
            power_density = integral_m2 / 10000  # Convert mW/m² to mW/cm²
        elif units == 'mW/m2':
            power_density = integral_m2
        elif units == 'W/cm2':
            power_density = integral_m2 / 10000000  # Convert mW/m² to W/cm²
        elif units == 'W/m2':
            power_density = integral_m2 / 1000  # Convert mW/m² to W/m²
        else:
            raise ValueError(f"Unsupported units: {units}. Supported units: 'mW/cm2', 'mW/m2', 'W/cm2', 'W/m2'")
        
        return power_density
    
    def calculate_exposure_time(self, spectral_df, wavelength_start, wavelength_end, target_energy, filter_df=None, units='J/cm2'):
        """
        Calculate exposure time needed to reach a given energy density level.
        
        Parameters:
        -----------
        spectral_df : pd.DataFrame
            DataFrame with columns 'wavelength' and 'intensity'
        wavelength_start : float
            Starting wavelength for integration (nm)
        wavelength_end : float
            Ending wavelength for integration (nm)
        target_energy : float
            Target energy density to reach
        filter_df : pd.DataFrame, optional
            DataFrame with columns 'wavelength' and 'attenuation' representing filter values.
            If provided, the spectral intensity will be divided by attenuation values before integration.
            Default is None (no filter applied).
        units : str, optional
            Energy density units. Options: 'J/cm2' (default), 'J/m2'
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'time_seconds': exposure time in seconds
            - 'time_minutes': exposure time in minutes
            - 'power_density': power density in W/cm² or W/m² (depending on units)
            - 'units': energy density units used
        """
        # Calculate power density in W/cm² or W/m²
        if units == 'J/cm2':
            power_density = self.calculate_power_density(spectral_df, wavelength_start, wavelength_end, filter_df, units='W/cm2')
        elif units == 'J/m2':
            power_density = self.calculate_power_density(spectral_df, wavelength_start, wavelength_end, filter_df, units='W/m2')
        else:
            raise ValueError(f"Unsupported units: {units}. Supported units: 'J/cm2', 'J/m2'")
        
        # Calculate exposure time: Time = Energy / Power
        if power_density <= 0:
            raise ValueError("Power density must be positive to calculate exposure time.")
        
        time_seconds = target_energy / power_density
        time_minutes = time_seconds / 60
        
        return {
            'time_seconds': time_seconds,
            'time_minutes': time_minutes,
            'power_density': power_density,
            'units': units
        }
    
    def _get_photopic_luminosity_function(self, wavelengths):
        """
        Get photopic luminosity function V(λ) values for given wavelengths.
        Uses CIE 1931 standard observer photopic luminosity function.
        
        Parameters:
        -----------
        wavelengths : array-like
            Wavelengths in nm
            
        Returns:
        --------
        np.ndarray
            V(λ) values (dimensionless, 0-1)
        """
        # CIE 1931 photopic luminosity function (V(λ)) - key wavelengths
        # Standard values from 380nm to 780nm at 5nm intervals
        cie_wavelengths = np.arange(380, 785, 5)
        cie_v_lambda = np.array([
            0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040, 0.0073,
            0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600, 0.0739, 0.0910, 0.1126,
            0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932,
            0.8620, 0.9149, 0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
            0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412, 0.3810, 0.3210,
            0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816, 0.0610, 0.0446, 0.0320, 0.0232,
            0.0170, 0.0119, 0.0082, 0.0057, 0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007,
            0.0005, 0.0004, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000,
            0.0000
        ])
        
        # Interpolate V(λ) values for the given wavelengths
        v_lambda = np.interp(wavelengths, cie_wavelengths, cie_v_lambda)
        
        return v_lambda
    
    def power_to_lux(self, spectral_df, filter_df=None):
        """
        Convert power spectrum to illuminance in lux.
        
        Lux is calculated by integrating the spectral power distribution weighted by
        the photopic luminosity function V(λ) and multiplying by 683 lm/W.
        
        Formula: Lux = ∫ V(λ) * P(λ) dλ * 683
        
        Where:
        - V(λ) is the photopic luminosity function (CIE 1931)
        - P(λ) is the spectral power distribution (W/m²/nm)
        - 683 is the conversion factor from watts to lumens
        
        Parameters:
        -----------
        spectral_df : pd.DataFrame
            DataFrame with columns 'wavelength' and 'intensity'
            Intensity should be in mW/m²/nm
        filter_df : pd.DataFrame, optional
            DataFrame with columns 'wavelength' and 'attenuation' representing filter values.
            If provided, the spectral intensity will be divided by attenuation values before conversion.
            Default is None (no filter applied).
            
        Returns:
        --------
        float
            Illuminance in lux (lm/m²)
        """
        if spectral_df is None or len(spectral_df) == 0:
            raise ValueError("No spectral data provided.")
        
        if 'wavelength' not in spectral_df.columns or 'intensity' not in spectral_df.columns:
            raise ValueError("DataFrame must contain 'wavelength' and 'intensity' columns.")
        
        # Apply filter if provided
        if filter_df is not None:
            spectral_df = self.remove_filter(spectral_df, filter_df)
        
        # Get wavelengths and intensities
        wavelengths = spectral_df['wavelength'].values
        intensities = spectral_df['intensity'].values  # mW/m²/nm
        
        # Convert intensities from mW/m²/nm to W/m²/nm
        intensities_w = intensities / 1000
        
        # Get photopic luminosity function V(λ)
        v_lambda = self._get_photopic_luminosity_function(wavelengths)
        
        # Multiply spectral power by V(λ)
        weighted_power = intensities_w * v_lambda
        
        # Integrate over wavelength: ∫ V(λ) * P(λ) dλ
        integral = np.trapz(weighted_power, wavelengths)
        
        # Convert to lux: multiply by 683 lm/W
        lux = integral * 683
        
        return lux
    

