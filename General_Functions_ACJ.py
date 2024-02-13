import numpy as np 
import matplotlib.pyplot as plt 
import lmfit as lm 
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, LinearModel, ExponentialModel
import os 
import h5py as h5 

def normalize_data(arr, by_mean=False, id1=None, id2=None):
    """ Normalize ydata in an array/list to go from 0 to 1 
    
    Args: 
        arr (list/np.ndarray): array to normalize
    
    Returns: 
        (np.ndarray) Scaled arr on range [0,1]
    """
    arr = np.array(arr)
    if by_mean:  
         return (arr-np.mean(arr[id1:id2]))/(np.max(arr)-np.mean(arr[id1:id2]))
    else: 
        return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def find_idx(wv, arr): 
    """
    Find the closest index for a specific wavenumber/wavelength/energy in an array of wavenumber/wavelength/energies 

    Args: 
        wv (int or float): Value you want the index of 
        arr (np.ndarray): Array of wavenumber/wavelength/energies

    Returns: 
        i (int): the nearest index
    """
    i = (np.abs(arr - wv)).argmin()
    
    return i 

def wv_range(x, wv1, wv2):
    """
    For a given array of energies, find index closest to specific start/stop points 

    Args: 
        x (np.ndarray) : array of energies 
        wv1 (float or int) : first energy in range  
        wv2 (float or int) : last energy in range 
    
    Returns: 
        start, stop (int): indices in x to get nearest to wv1, wv2 
    """
    start = np.argmin(np.abs(x - wv1))
    stop = np.argmin(np.abs(x - wv2))
    return start, stop 

def custom_function( x:np.ndarray, y:np.ndarray, npeaks:int, peakfunction:lm.models, backgroundfunction:lm.models, centers:list | np.ndarray, peaktol:float | int=100, diffpeaks:bool=False):
        """"
        Function to build custom lmfit model for an arbitrary spectra given spectra data, number of peaks to fit, 
        function to fit background, function to fit peaks (currently all peaks fit with same function)

        Args: 
            x: xdata / energies 
            y: ydata / spectra to fit 
            npeaks (int): Number of peaks to fit 
            peakfunction (lmfit.models): Model to fit all peaks, typically LorenzianModel, GaussianModel, VoigtModel 
            backgroundfuction (lmfit.models): Model to fit background, typically ExponentialModel, LinearModel
            centers (list): Initial guess for peak centers
            peaktol (int or float): Min/max range for peak center in fitting 
            diffpeaks (bool): If you want to fit each peak to specific lorentzian/gaussian/voigt model
        
        Returns: 
            out (lmfit.model.ModelResult): Model result 
        """

        bg_pre_dir = {ExponentialModel:'bgexp_', LinearModel:'bglin_'}
        model = backgroundfunction(prefix=bg_pre_dir[backgroundfunction])
        pars = model.guess(y, x=x)

        if diffpeaks == False: 
            pre_dir = {ExponentialModel:'exp', GaussianModel:'g', LorentzianModel:'l', VoigtModel:'v'}
            pre = pre_dir[peakfunction]

            for n in np.arange(npeaks): 
                mod = peakfunction(prefix=f'{pre}{n}_')
                init_center = centers[n]

                pars += mod.guess(y, x=x, center=init_center)
                pars[f'{pre}{n}_amplitude'].min = 0
                pars[f'{pre}{n}_center'].min = init_center - peaktol
                pars[f'{pre}{n}_center'].max = init_center + peaktol
                # other constraints 
                model += mod

        out = model.fit(y, pars, x=x)
        return out 

def create_sample_dict(data_dir, date, settings_i_want=[]): 
    sorted_files = os.listdir(data_dir + date)
    sorted_files = [f for f in sorted_files if '.tif' in f or '.h5' in f]
    sorted_files.sort()
    sample_dict = {}

    for f in sorted_files: 
        fn = f'{data_dir}{date}/{f}' 
        time = fn.split('/')[-1].split('_')[1]
        # date = fn.split('/')[-1].split('_')[0]
        measurement = '_'.join(fn.replace('.h5', '').split('/')[-1].split('_')[2:])
        if 'h5' in f: 
            try: 
                file = h5.File(fn)
                name = dict(file['app/settings'].attrs.items())['sample']
                # if measurement == 'thor_cam_capture': 
                #     settings_i_want = ['axis_3_position', 'axis_4_position']
                # if measurement == 'time_sweep': 
                #     settings_i_want = ['mod_freq', 'time_constant', 'filter_order', 'lockin_read_time', 'switch_time', 'end_time', 'step_time_high', 'step_time_low',
                #         'axis_3_position', 'axis_4_position']
                if name not in sample_dict.keys(): 
                    sample_dict[name] = {fn:{'time':time, 'measurement':measurement, 'settings' : extract_h5_settings_HiP(file, settings_i_want, measurement)}}
                else: 
                    sample_dict[name][fn] = {'time':time, 'measurement':measurement, 'settings' : extract_h5_settings_HiP(file, settings_i_want, measurement)}
            except OSError: 
                pass 
    return sample_dict

def extract_h5_settings_HiP(file, settings_i_want, measurement): 
    """
    Extract desired settings from an h5 file 
    """
    settings = {}

    all_settings = dict(file[f'measurement/{measurement}/settings'].attrs.items())

    # TDTR_Lockin_settings = dict(file['hardware/TDTR_Lockin/settings'].attrs.items())
    # picomotor_settings = dict(file['hardware/picomotor/settings'].attrs.items())
    # all_settings.update(TDTR_Lockin_settings)
    # all_settings.update(picomotor_settings)

    for i_want in settings_i_want: 
        settings[i_want] = all_settings[i_want]
    return settings

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]