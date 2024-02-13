import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, GaussianModel, VoigtModel, LorentzianModel, ExponentialModel
import tqdm
import scipy
import pandas as pd 
from General_Functions_ACJ import *


def to_wv(ex, arr): 
    """ Convert nm to relative wavenumbers given an excitation wavelength and array of energies in nm

    Args: 
        ex (int or float): excitation wavelength in nm 
        arr (list or np.ndarray or pd.Series) : array of energies in nm to be converted to cm-1
    
    Returns: 
        (np.ndarray) : array of converted wavelengths 
    """

    arr = np.array(arr)
    return 10**7/ex - 10**7/arr

class raman_map_data: 
    """ 
    A class to analyze, plot, and fit Raman/PL data from Horiba LabRam and XPloRa and Liu Lab Confocal Raman/PL
    
    Args: 

    Attributes:
        params (dict): recorded acquisition parameters
        name (str): filename 
        offset (int or float): arbitrary plotting offset 
        energies (np.ndarray): xdata of wavelengths/energies
        data (np.ndarray): ydata/spectra from different positions in map  
        pos (np.ndarray): len(data) x 2 array of map positions
        av_data (np.ndarray): average ydata/spectra over whole map 
        unphysical (np.ndarray): indices corresponding to unphysical spectra in map 

    Methods:
        wv_range: 
        find_fitted_peak_av: 
        fit_.....: 
        find_idx: 
        plot_map: 
        threshold: 
    """
    def __init__(self, fn, add_params=[], print_params=True, bkg=None, tool='Liu', raman=True, moke=False, extended=False, laser=532.5):
#         path = 'G:My Drive/Raman_Data/'
        params =['#Acq. time (s)=\t', '#Accumulations=\t', '#Grating=\t', '#ND Filter=\t', '#Laser=\t', '#Hole=\t']
        params.extend(add_params)
        self.params = {}
        self.name = fn 
        self.offset = 0
        if fn == '': 
            pass   

        else: 
            if tool == 'LabRam' or tool == 'XPlora':
                with open(fn, errors='ignore') as f:
                    param_num = 0
                    i = 0
                    num_lines = 0
                    for line in f: 
                        if line.startswith(tuple(params)): 
                            txt = line.replace('#', '').split('=')
                            name = txt[0]
                            value = txt[1].replace('\t', '').replace('\n', '')
                            if print_params == True:
                                print(name,':', value)
                            self.params[name] = value
                        if line.startswith('#'):
                            param_num += 1
                        if '#' not in line and i < 1: 
                            energies = [float(i) for i in line.split()]
                            i +=1
                        num_lines +=1 
                data_dim = num_lines - param_num - 1

                with open(fn, errors='ignore') as f:
                    data = np.zeros([data_dim, len(energies)])
                    pos = np.zeros([data_dim, 2])
                    i = 0
                    j = 0
                    for line in f: 
                        if '#' not in line: 
                            if j == 0 and i == 0: 
                                j += 1
                            else: 
                                data[i] = [float(counts) for counts in line.split()[2:]]
                                pos[i] = [float(position) for position in line.split()[:2]]
                                i +=1

            elif tool == 'Liu': 
                if extended == True: 
                    temp = pd.read_csv(fn, header=1)
                    num_points = temp.shape[1]
                    wv1 = float(input('beginning of spectral range'))
                    wv2 = float(input('end of spectral range'))

                    cols =  ['Scan', 'Step', 'X index', 'Y index', 'X (um)', 'Y (um)', 'Vx (um)',
       'Vy (um)'] + list(np.linspace(wv1, wv2, num_points-8))
                    s = pd.read_csv(fn, header=0, names=cols)
                    data = np.array(s[s.columns[8:]])
                    energies = np.linspace(wv1, wv2, num_points-8)
                else: 
                    s = pd.read_csv(fn)
                    wavelengths = [col for col in s.columns if 'nm' in col]
                    data = np.array(s[wavelengths])
                    energies = np.array([float(name.split(' ')[0]) for name in wavelengths])
                if raman == True: 
                    energies  = to_wv(laser, energies)

                pos = [[float(y), float(x)] for x, y in zip(s['X (um)'], s['Y (um)'])]
                pos = np.array(pos)

                if moke == True: 
                    pos = np.unique(pos, axis=0)
                    roi1_id = np.arange(0, data.shape[0], 2)
                    roi2_id = np.arange(1, data.shape[0], 2)
                    self.roi1_data = data[roi1_id]
                    self.roi2_data = data[roi2_id]
            
            if type(bkg) != None: 
                self.bg = bkg
            
            self.energies = np.array(energies)
            self.data = data 
            self.pos = pos
            self.av_data = self.data.mean(axis=0)
            self.unphysical = np.array([])

    def wv_range(self, wv1, wv2):
        """
        In self.energies, find index closest to specific start/stop points 

        Args: 
            wv1 (float or int) : first energy in range  
            wv2 (float or int) : last energy in range 
        
        Returns: 
            start, stop (int): indices in self.energies to get nearest to wv1, wv2 
        """
        start = np.argmin(np.abs(self.energies - wv1))
        stop = np.argmin(np.abs(self.energies - wv2))
        return start, stop 
    
    def calibrate_wv(self,si_guess=520): 
        start = 475 
        stop = 600 
        if self.energies.max() < stop: 
            stop = self.energies.max()

        if self.energies.min() > start: 
            stop = self.energies.min()

        start, stop = self.wv_range(start, stop)
        x = self.energies[start:stop]
        y = self.av_data[start:stop]

        out = self.custom_function(x=x, y=y, npeaks=1, peakfunction=LorentzianModel, backgroundfunction=LinearModel, centers=[si_guess], peaktol=20)

        si = out.params['l0_center'].value
        calibration = 520 - si 

        if calibration > 0: 
            calibrated_energies = self.energies - calibration

        if calibration < 0: 
            calibrated_energies = self.energies + calibration

        self.raw_energies  = self.energies
        self.energies = calibrated_energies
    
    def find_idx(self, energy):
        return np.absolute(self.energies - energy).argmin()
    
    def custom_function(self, x:np.ndarray, y:np.ndarray, npeaks:int, peakfunction:lm.models, backgroundfunction:lm.models, centers:list | np.ndarray, peaktol:float | int=100, diffpeaks:bool=False):
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
    
    def fit_map(self, wv_rng:tuple, npeaks, pars=['center', 'amplitude', 'fwhm'], use_background:bool=False, change_data = None, print_it=True, **kwargs):
        """
        
        """

        # establish fitting range 
        wv1, wv2 = wv_rng
        start, stop = self.wv_range(wv1, wv2) 
        xtemp = self.energies[start:stop]

        # set up fit storage 

        if type(change_data) == None: 
            data = self.data 
        else: 
            data = change_data

        self.map_fits = np.zeros([len(data), 3, len(xtemp)])
        self.map_fits[:,:1] = xtemp

        y_uniq = (np.unique(self.pos[:,1]))
        x_uniq = (np.unique(self.pos[:,0]))

        x_coord, y_coord = np.meshgrid(x_uniq, y_uniq)

        peak_pars = [p for p in pars if 'bg' not in p]
        bg_pars = [p for p in pars if 'bg' in p]
        npars = len(peak_pars)
        maps = np.zeros([npars, npeaks*2, len(y_uniq), len(x_uniq)])
        bg_maps = np.zeros([len(bg_pars), len(y_uniq), len(x_uniq)])

        peakfunction = kwargs['peakfunction']
        pre_dir = {ExponentialModel:'exp', GaussianModel:'g', LorentzianModel:'l', VoigtModel:'v'}
        pre = pre_dir[peakfunction]


        # fit data 
        for i, ytemp in (enumerate(data)):
            x, y = self.pos[i]
            xnew, ynew = np.argwhere((x_coord == x) & (y_coord == y))[0]
            if use_background:
                ytemp = ytemp - self.bg
            try:     
                ytemp = ytemp[start:stop]
                out = self.custom_function(x=xtemp, y=ytemp, npeaks=npeaks, **kwargs)
                self.map_fits[i, 1] = ytemp
                self.map_fits[i, 2] = out.best_fit
                for param_id, param_name in enumerate(peak_pars): 
                    for n in np.arange(npeaks): 
                        full_parameter_name = pre + str(n) + '_' + param_name
                        try: 
                            fitted_param = out.params[full_parameter_name].value
                            err = out.params[full_parameter_name].stderr 
                            maps[param_id][n][xnew, ynew] = fitted_param
                            maps[param_id][n+npeaks][xnew, ynew] = err  
                        except KeyError: 
                            print(f'you spelled something wrong. {full_parameter_name} not in fit result parameters')
                            pass  
                for param_id, param_name in enumerate(bg_pars): 
                    full_parameter_name =  param_name
                    try: 
                        fitted_param = out.params[full_parameter_name].value
                        err = out.params[full_parameter_name].stderr 
                        bg_maps[param_id][xnew, ynew] = fitted_param
                        bg_maps[param_id][xnew, ynew] = err  
                    except KeyError: 
                        print(f'you spelled something wrong. {full_parameter_name} not in fit result parameters')
                        pass   
                
            except ValueError: 
                self.map_fits[i, 1] = np.NaN
                self.map_fits[i, 2] = np.NaN
                for param_id, param_name in enumerate(pars): 
                    for n in np.arange(npeaks): 
                        fitted_param = np.NaN
                        err = np.NaN
                        maps[param_id][n][xnew, ynew] = fitted_param
                        maps[param_id + npeaks][n][xnew, ynew] = err  
        self.fitted_maps = maps 
        self.fitted_bgs = bg_maps
        print('Fitting Completed \n')
        if print_it == True: 
            for i, par in enumerate(pars): 
                print(f'fits for parameter parameter \"{par}\" at index {i} of fitted_map')
            print(f'\nexample: if npeaks = {npeaks}, the map for the fitted {pars[0]} of peak {npeaks} is at fitted_maps[0, {npeaks-1}]; \n\t the map of fitted errors is at fitted_maps[0, {npeaks - 1 + npeaks}] ')
    
    def raw_inten_map(self, ranges:np.ndarray, change_data = None, moke=False): 
        """
        No fitting. Map of Integrated Intensity 
        """
        if type(change_data) == None:  
            data = self.data 
        else: 
            data = change_data

        y_uniq = (np.unique(self.pos[:,1]))
        x_uniq = (np.unique(self.pos[:,0]))
        x_coord, y_coord = np.meshgrid(x_uniq, y_uniq)
        
        maps = np.zeros([len(ranges), len(y_uniq), len(x_uniq)])
        for i, range in enumerate(ranges): 
            e1, e2 = range 
            range_slice = slice(self.find_idx(e1), self.find_idx(e2))
            for j, ytemp in enumerate(data):
                x, y = self.pos[j]
                xnew, ynew = np.argwhere((x_coord == x) & (y_coord == y))[0]
                maps[i][xnew, ynew] += np.nansum(list(map(float, ytemp[range_slice])))
            if change_data == False and moke == True: 
                maps[i] = maps[i] / 2
                print('moke data, returning average map between 2 rois')
            maps[i] = normalize_data(maps[i])
        
        self.inten_maps = maps 

    def threshold_map(self, m, map_type, threshold_low, threshold_high, show_plot=True): 
        px = m.shape[0]
        py = m.shape[1]
        self.thresholded_out = np.array([])

        if map_type == 'amplitude': 
            thresholded_out = m < 0 
            thresholded_out_i = np.argwhere(m.ravel() < 0) 
        elif map_type == 'peak_location':
            thresholded_out = np.logical_or(m < threshold_low, m > threshold_high)
            thresholded_out_i = np.argwhere(np.logical_or(m.ravel() < threshold_low, m.ravel() > threshold_high))
        elif map_type == 'ratio' or map_type == 'diff':
            pass

        where_threshold = np.append(np.argwhere(thresholded_out), np.argwhere((np.isnan(m))), axis=0)

        thresholded = m.copy()
        for i in where_threshold: 
            id_x, id_y = i 
            thresholded[id_x, id_y] = np.nanmedian(m)
        if show_plot: 
            f, ax = plt.subplots(1,2)
            im1 = ax[1].imshow(thresholded)
            f.colorbar(im1, orientation='vertical')
            ax[1].set_title('thresholded map')
            
            im0 = ax[0].imshow(m)
            f.colorbar(im0, orientation='vertical')
            ax[0].set_title('original map')
            self.thresholded = thresholded
            self.thresholded_out = np.append(self.thresholded_out, thresholded_out_i)
            return ax 
        self.thresholded = thresholded
        self.thresholded_out = np.append(self.thresholded_out, thresholded_out_i)
    
    def flatten_threshold_map(self, m, map_type, threshold_low, threshold_high, quad=True, show_threshold_plot=False, show_plot=True): 
        px = m.shape[0]
        py = m.shape[1]
        
        ax = self.threshold_map(m, map_type, threshold_low, threshold_high, show_plot=False)

        xx, yy = np.meshgrid(np.arange(py), np.arange(px))


        if quad: 
            xy = np.array([xx.ravel(), yy.ravel()]).T
            A = np.c_[np.ones(len(self.thresholded.ravel())), xy, np.prod(xy, axis=1), xy**2]
            C,_,_,_ = scipy.linalg.lstsq(A, self.thresholded.ravel())

            # evaluate it on a grid
            x = xx.flatten()
            y = yy.flatten()
            Z = np.dot(np.c_[np.ones(x.shape), x, y, x*y, x**2, y**2], C).reshape(xx.shape)
        else: 

            A = np.c_[xx.ravel(), yy.ravel(), np.ones_like(self.thresholded.ravel())]
            C,_,_,_ = scipy.linalg.lstsq(A, self.thresholded.ravel())

            Z = C[0]*xx + C[1]*yy + C[2] 

        if show_plot: 

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection':'3d'}, figsize=(30,10))

            surf = ax1.plot_surface(xx, yy, (self.thresholded), linewidth=0, antialiased=False)
            ax1.set_title('Thresholded Map Using For Fit')
            ax2.plot_surface(xx, yy, Z)
            ax2.plot_surface(xx, yy, self.thresholded)
            ax2.set_title('Fitting Plane')
            ax3.plot_surface(xx, yy, self.thresholded - Z)
            ax3.set_title('Subtracting Fitted Plane From Raw Data')

        self.flattened = self.thresholded - Z
        
#         if bkg != None: 
#             bg_name = path + bkg
#             dat_bg = pd.read_csv(bg_name, encoding='unicode_escape', header=self.header, delimiter='\t',names=['cm-1', 'counts'])
#             self.bg = np.array(dat_bg['counts'])
#         else: 
#             self.bg = None
        

#     def plot(self, ax,offset=0, bg_subtract = True, plot_bg=False, **kwargs):
#         self.offset = offset
#         ax.set_xlabel('cm$^{-1}$')
#         ax.set_ylabel('counts') 
#         if type(self.bg) != type(None) and bg_subtract == True: 
#             ax.plot(self.x, self.y + offset - self.bg, **kwargs)
#             print('background subtracted')
#         else: 
#             ax.plot(self.x, self.y + offset, **kwargs)
#         if type(self.bg) != type(None) and plot_bg == True: 
#             ax.plot(self.x, self.bg + offset, label='background', **kwargs)

#     def find_absolute_peak(self, wv1, wv2, plot = True, ax=None):
#         start, stop = self.wv_range(wv1, wv2)
#         peak_y = np.max(self.y[start:stop])
#         id_peak = np.argmax(self.y[start:stop])
#         peak_x = self.x[id_peak + start]
#         if plot: 
# #             self.plot(ax)/
#             ax.plot(peak_x, peak_y, 'k*')
#             print(f'peak @ {peak_x} cm-1')

#     def find_fitted_peak_av(self, wv1, wv2, center=None, func='gaussian', composite=ExponentialModel(prefix='e1_'), plot=True, ax=None, **kwargs): 
#         func_dict = {'gaussian':GaussianModel(prefix='g1_'), 'voigt':VoigtModel(prefix='v1_'), 'lorentzian':LorentzianModel(prefix='l1_')}
#         start, stop = self.wv_range(wv1, wv2)
#         xnew = self.energies[start:stop]
#         mod = func_dict[func] + composite


# #         if type(self.bg) != type(None): 
# #             ynew = self.av_data - self.bg
# #         else: 
#         ynew = self.av_data
        
            
#         ynew = ynew[start:stop]
#         if func == 'lorentzian': 
#             mod.set_param_hint('l1_amplitude', min=0)
#             if center != None: 
#                 mod.set_param_hint('l1_center', value=center)
#         pars = mod.make_params()
# #         print(mod.prefix)
        
#         msk = [param.endswith('center') for param in pars.keys()]
#         peak_key = np.array(list(pars.keys()))[msk][0]
        
        
#         out = mod.fit(ynew, pars, x=xnew)
#         peak = out.values[peak_key]
#         print('peak:', peak, 'cm-1')
#         if plot: 
# #             self.plot(ax, **kwargs)
#             ax.plot(xnew, out.best_fit + self.offset, **kwargs)
#         return peak, out
    
#     def fit_lor_lor_exp(self, wv1, wv2, center, y  = None):
            
#         start, stop = self.wv_range(wv1, wv2)
#         x = self.energies[start:stop]
#         if type(y) == None: 
#             y = self.av_data

#         y = y[start:stop]
        
#         peak1, peak2 = center
#         lor_mod1 = LorentzianModel(prefix='lor1_')
#         lor_mod2 = LorentzianModel(prefix='lor2_')
#         exp_mod = ExponentialModel(prefix='exp_')

#         pars = exp_mod.guess(y, x=x)
#         pars += lor_mod1.guess(y, x=x, center=peak1)
#         pars += lor_mod2.guess(y, x=x, center=peak2)

#         mod = lor_mod1 + exp_mod + lor_mod2
#         out = mod.fit(y, pars, x=x)
#         return x, y, out

#     def fit_3lor_exp(self, wv1, wv2, center, y  = None):
            
#         start, stop = self.wv_range(wv1, wv2)
#         x = self.energies[start:stop]
#         if type(y) == None: 
#             y = self.av_data
#         y = y[start:stop]
        
#         peak1, peak2, peak3 = center
#         lor_mod1 = LorentzianModel(prefix='lor1_')
#         lor_mod2 = LorentzianModel(prefix='lor2_')
#         lor_mod3= LorentzianModel(prefix='lor3_')
#         exp_mod = ExponentialModel(prefix='exp_')

#         pars = exp_mod.guess(y, x=x)
#         pars += lor_mod1.guess(y, x=x, center=peak1)
#         pars += lor_mod2.guess(y, x=x, center=peak2)
#         pars += lor_mod3.guess(y, x=x, center=peak3)

#         mod = lor_mod1 + exp_mod + lor_mod2 + lor_mod3
#         out = mod.fit(y, pars, x=x)
#         return x, y, out
    
#     def fit_lor_exp(self, wv1, wv2, center, y  = None):
#         if type(y) == None: 
#             y = self.av_data
            
#         start, stop = self.wv_range(wv1, wv2)
#         x = self.energies[start:stop]
        
            
#         y = y[start:stop]
        
#         peak1, peak2 = center
#         lor_mod1 = LorentzianModel(prefix='lor1_')
#         exp_mod = ExponentialModel(prefix='exp_')

#         pars = exp_mod.guess(y, x=x)
#         pars += lor_mod1.guess(y, x=x, center=peak1)
        
#         pars['lor1_amplitude'].max = self.av_data[self.find_idx(peak1)]+ 100
#         pars['lor1_amplitude'].min = self.av_data[self.find_idx(peak1)] - 100
        
#         pars['exp_amplitude'].min = 0
        
#         mod = lor_mod1 + exp_mod
#         out = mod.fit(y, pars, x=x)
#         return x, y, out
    
#     def fit_gaus_exp(self, wv1, wv2, center, y  = None):
#         if type(y) == None: 
#             y = self.av_data
            
#         start, stop = self.wv_range(wv1, wv2)
#         x = self.energies[start:stop]
        
            
#         y = y[start:stop]
        
#         peak1, peak2 = center
#         lor_mod1 = GaussianModel(prefix='lor1_')
#         exp_mod = ExponentialModel(prefix='exp_')

#         pars = exp_mod.guess(y, x=x)
#         pars += lor_mod1.guess(y, x=x, center=peak1)
        
#         pars['lor1_amplitude'].max = self.av_data[self.find_idx(peak1)]+ 100
#         pars['lor1_amplitude'].min = self.av_data[self.find_idx(peak1)] - 100
#         # pars['lor1_center'].max = 46
#         # pars['lor1_center'].min = 36

#         pars['exp_amplitude'].min = 0
        
#         mod = lor_mod1 + exp_mod
#         out = mod.fit(y, pars, x=x)
#         return x, y, out
    
#     def fit_gaus_lin(self, wv1, wv2, center, y  = None):
#         if type(y) == None: 
#             y = self.av_data
            
#         start, stop = self.wv_range(wv1, wv2)
#         x = self.energies[start:stop]
        
            
#         y = y[start:stop]
        
#         peak1, peak2 = center
#         lor_mod1 = GaussianModel(prefix='lor1_')
#         lin_mod = LinearModel()

#         pars = lin_mod.guess(y, x=x)
#         pars += lor_mod1.guess(y, x=x, center=peak1)
        
#         # pars['lor1_amplitude'].max = self.av_data[self.find_idx(peak1)]+ 100
#         # pars['lor1_amplitude'].min = self.av_data[self.find_idx(peak1)] - 100
#         # pars['lor1_center'].max = 46
#         # pars['lor1_center'].min = 36

        
#         mod = lor_mod1 + lin_mod
#         out = mod.fit(y, pars, x=x)
#         return x, y, out
   
    # def plot_map(self, wv_rng=[], map_type='intensity_summed', map1=[383, 386], map2=[403, 406], map3=[175, 620], show_plot=True, y=None, *kwargs):

    #     start, stop = self.wv_range(wv_rng[0], wv_rng[1])
    #     x = self.energies[start:stop]
    #     self.map_fits = np.zeros([len(self.data), 3, len(x)])
    #     self.map_fits[:,:1] = x
        
    #     y_uniq = (np.unique(self.pos[:,1]))
    #     x_uniq = (np.unique(self.pos[:,0]))

    #     x_coord, y_coord = np.meshgrid(x_uniq, y_uniq)
    #     m1 = np.zeros([len(y_uniq), len(x_uniq)])
    #     m2 = np.zeros([len(y_uniq), len(x_uniq)])
    #     m3 = np.zeros([len(y_uniq), len(x_uniq)])
        
    #     if map_type == 'intensity_summed': 
    #         e1_map1, e2_map1 = map1
    #         e1_map2, e2_map2 = map2
    #         e1_map3, e2_map3 = map3
            
    #         range_1 = slice(self.find_idx(e1_map1), self.find_idx(e2_map1))
    #         range_2 = slice(self.find_idx(e1_map2), self.find_idx(e2_map2))
    #         range_3 = slice(self.find_idx(e1_map3), self.find_idx(e2_map3))
        
    #         for i, val in enumerate(self.data):
    #             x, y = self.pos[i]
    #             xnew, ynew = np.argwhere((x_coord == x) & (y_coord == y))[0]
    #             m1[xnew, ynew] += np.nansum(list(map(float, val[range_1])))
    #             m2[xnew, ynew] += np.nansum(list(map(float, val[range_2])))
    #             m3[xnew, ynew] += np.nansum(list(map(float, val[range_3])))
            
    #         m1 = normalize_data(m1)
    #         m2 = normalize_data(m2)
    #         m3 = normalize_data(m3)
            
    #         self.inten1_map = m1
    #         self.inten2_map = m2
    #         self.inten3_map = m3
    #         if show_plot: 
    #             f, (axes) = plt.subplots(1,3, figsize=(12,5))
    #             (ax1, ax2, ax3) = axes
    #             ax1.imshow(m1, cmap='Reds')
    #             ax1.set_title(fr'intensity, ~{e1_map1} - {e2_map1} $cm^{-1}$')
    #             ax2.imshow(m2, cmap='Blues')
    #             ax2.set_title(fr'intensity, ~{e1_map2} - {e2_map2} $cm^{-1}$')
    #             ax3.imshow(m3, cmap='Greens')
    #             ax3.set_title(fr'intensity, ~{e1_map3} - {e2_map3} $cm^{-1}$')

    #     elif map_type == 'fit_peaks':
    #         m4 = np.zeros([len(y_uniq), len(x_uniq)])
    #         m5 = np.zeros([len(y_uniq), len(x_uniq)])
    #         m6 = np.zeros([len(y_uniq), len(x_uniq)])
            
    #         peak1_list = []
    #         peak2_list = []
    #         amp1_list = []
    #         amp2_list = []
    #         ratio_list = []
    #         diff_list = []
            
    #         for i, val in (enumerate(self.data)):
    #             x, y = self.pos[i]
    #             xnew, ynew = np.argwhere((x_coord == x) & (y_coord == y))[0]
    #             if type(y) == None: 
    #                 pass
    #             elif y == 'bkg': 
    #                 val = val - self.bg
    #             try: 
    #                 if fit_func == 'fit_lor_lor_exp':
                         
    #                     x_spec, y_spec, out = self.fit_lor_lor_exp(wv1, wv2, center, y=val)
    #                     self.map_fits[i, 1] = y_spec
    #                     self.map_fits[i, 2] = out.best_fit

    #                     peak1 = round(out.params['lor1_center'].value, 3)
    #                     peak1_err1 = out.params['lor1_center'].stderr

    #                     peak2 = round(out.params['lor2_center'].value, 3)
    #                     peak2_err2 = out.params['lor2_center'].stderr

    #                     amp1 = round(out.params['lor1_amplitude'].value, 3)
    #                     amp1_err1 = out.params['lor1_amplitude'].stderr

    #                     amp2 = round(out.params['lor2_amplitude'].value, 3)
    #                     amp2_err2 = out.params['lor2_amplitude'].stderr

    #                 elif fit_func == 'fit_lor_exp': 
    #                     x_spec, y_spec, out = self.fit_lor_exp(wv1, wv2, center, y=val)
    #                     self.map_fits[i, 1] = y_spec
    #                     self.map_fits[i, 2] = out.best_fit

    #                     peak1 = round(out.params['lor1_center'].value, 3)
    #                     peak1_err1 = out.params['lor1_center'].stderr

    #                     amp1 = round(out.params['lor1_amplitude'].value, 3)
    #                     amp1_err1 = out.params['lor1_amplitude'].stderr 
                    
    #                 elif fit_func == 'fit_gaus_exp': 
    #                     x_spec, y_spec, out = self.fit_gaus_exp(wv1, wv2, center, y=val)
    #                     self.map_fits[i, 1] = y_spec
    #                     self.map_fits[i, 2] = out.best_fit

    #                     peak1 = round(out.params['lor1_center'].value, 3)
    #                     peak1_err1 = out.params['lor1_center'].stderr

    #                     amp1 = round(out.params['lor1_amplitude'].value, 3)
    #                     amp1_err1 = out.params['lor1_amplitude'].stderr 

    #                 elif fit_func == 'fit_gaus_lin': 
    #                     x_spec, y_spec, out = self.fit_gaus_lin(wv1, wv2, center, y=val)
    #                     self.map_fits[i, 1] = y_spec
    #                     self.map_fits[i, 2] = out.best_fit

    #                     peak1 = round(out.params['lor1_center'].value, 3)
    #                     peak1_err1 = out.params['lor1_center'].stderr

    #                     amp1 = round(out.params['lor1_amplitude'].value, 3)
    #                     amp1_err1 = out.params['lor1_amplitude'].stderr 

    #                     peak2 = round(out.params['lor1_center'].value, 3)
    #                     peak2_err2 = out.params['lor1_center'].stderr

    #                     amp2 = round(out.params['lor1_amplitude'].value, 3)
    #                     amp2_err2 = out.params['lor1_amplitude'].stderr 
                    
    #             except ValueError: 
    #                 self.map_fits[i, 1] = np.NaN
    #                 self.map_fits[i, 2] = np.NaN
    #                 peak1 = np.NaN 
    #                 peak1_err1 = np.NaN 
                    
    #                 peak2 = np.NaN
    #                 peak2_err2 = np.NaN
                    
    #                 amp1 = np.NaN 
    #                 amp2_err2 = np.NaN 

    #             if fit_func == 'fit_lor_lor_exp':
    #                 if amp1 <= 0 or amp2 <= 0 or peak1 <= 300 or peak2 <= 300 or peak2 >= 500 or peak1 >= 500: 
    #                     self.unphysical = np.append(self.unphysical, i)
                        
    #                 if peak2 > peak1: 
    #                     diff = peak2 - peak1 
    #                     m1[xnew, ynew] = peak1 
    #                     m2[xnew, ynew] = peak2
    #                     m4[xnew, ynew] = amp1
    #                     m5[xnew, ynew] = amp2
    #                     ratio = amp2/amp1
    #                     peak1_list.append(peak1)
    #                     peak2_list.append(peak2)
    #                     amp1_list.append(amp1)
    #                     amp2_list.append(amp2)
    #                 elif peak1 > peak2: 
    #                     diff = peak1 - peak2 
    #                     m1[xnew, ynew] = peak2
    #                     m2[xnew, ynew] = peak1
    #                     m4[xnew, ynew] = amp2 
    #                     m5[xnew, ynew] = amp1 
    #                     ratio = amp1/amp2
    #                     peak1_list.append(peak2)
    #                     peak2_list.append(peak1)
    #                     amp1_list.append(amp2)
    #                     amp2_list.append(amp1)

    #                 m3[xnew, ynew] = diff
    #                 m6[xnew, ynew] = ratio
                

    #                 ratio_list.append(ratio)
    #                 diff_list.append(diff)

    #             elif fit_func == 'fit_lor_exp':
    #                 if amp1 <= 0 or peak1 <= 0: 
    #                     self.unphysical = np.append(self.unphysical, i)
                    
    #             elif fit_func == 'fit_gaus_exp' or fit_func == 'fit_gaus_lin':
    #                 if amp1 <= 0 or peak1 <= 0: 
    #                     self.unphysical = np.append(self.unphysical, i)
                        
    #                 m1[xnew, ynew] = peak1 
    #                 m4[xnew, ynew] = amp1
    #                 peak1_list.append(peak1)
    #                 amp1_list.append(amp1)
            
    #         self.peak1_map = m1
    #         self.peak2_map = m2
    #         self.diff_map = m3
    #         self.amp1_map = m4
    #         self.amp2_map = m5
    #         self.ratio_map = m6
            
    #         if show_plot: 
    #             f, axes = plt.subplots(2,3, figsize=(12,10))
    #             ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes
    #             im1 = ax1.imshow(m1, cmap='Reds')
    #             ax1.set_title(fr'E12g center, av = {round(np.nanmean(peak1_list), 3)} $cm^{-1}$')
    #             im2 = ax2.imshow(m2, cmap='Blues')
    #             ax2.set_title(fr'A2g center, av = {round(np.nanmean(peak2_list), 3)} $cm^{-1}$')
    #             im3 = ax3.imshow(m3, cmap='Greens')
    #             ax3.set_title(fr'A2g - E12g center, av = {round(np.nanmean(diff_list), 3)}$cm^{-1}$')

    #             im4 = ax4.imshow(m4, cmap='Oranges')
    #             ax4.set_title(fr'E12g amplitude, av = {round(np.nanmean(amp1_list), 3)} $cm^{-1}$')

    #             im5 = ax5.imshow(m5, cmap='Purples')
    #             ax5.set_title(fr'A2g amplitude, av = {round(np.nanmean(amp2_list), 3)} $cm^{-1}$')

    #             im6 = ax6.imshow(m5, cmap='Greys')
    #             ax6.set_title(fr'A2g/E12g amplitude, av = {round(np.nanmean(ratio_list), 3)} $cm^{-1}$')

    #             f.colorbar(im1, orientation='vertical')
    #             f.colorbar(im2, orientation='vertical')
    #             f.colorbar(im3, orientation='vertical')

    #             f.colorbar(im4, orientation='vertical')
    #             f.colorbar(im5, orientation='vertical')
    #             f.colorbar(im6, orientation='vertical')
            
    #             return axes 
    #         else: 
    #             return 
            
    

#power analysis

def find_wv(wv, arr): 
    i = (np.abs(arr - wv)).argmin()
    return i 

def fit_lor_exp(x, y, peak): 
    lor_mod = LorentzianModel(prefix='lor1_')
    exp_mod = ExponentialModel(prefix='exp_')

    pars = exp_mod.guess(y, x=x)
    pars += lor_mod.guess(y, x=x, center=peak)

    pars['lor1_amplitude'].max = self.av_data[self.find_idx(peak1)]+ 100
    pars['lor1_amplitude'].min = self.av_data[self.find_idx(peak1)] - 100
    
    pars['exp_amplitude'].min = 0

    mod = lor_mod + exp_mod
    out = mod.fit(y, pars, x=x)
    return out

def fit_lor_lor_exp(x, y, peak1, peak2): 
    lor_mod1 = LorentzianModel(prefix='lor1_')
    lor_mod2 = LorentzianModel(prefix='lor2_')
    exp_mod = ExponentialModel(prefix='exp_')

    pars = exp_mod.guess(y, x=x)
    pars += lor_mod1.guess(y, x=x, center=peak1)
    pars += lor_mod2.guess(y, x=x, center=peak2)

    mod = lor_mod1 + exp_mod + lor_mod2
    out = mod.fit(y, pars, x=x)
    return out

# 532 perc to mW 
maxpower532 = 18.2 #mW 100 LWD
perc_conv532 = {0.01:0.01, 0.1:0.1, 1:0.92, 2.5:3.06, 5:5.17, 10:10.5, 25:24.6, 50:50.6, 100:100}

maxpower633 = 4.2 #mW 100 LWD
perc_conv633 = {0.01:0.01, 0.1:0.08, 1:0.72, 2.5:2.32, 5:4.55, 10:8.84, 25:22.1, 50:49.8, 100:100}


def get_params(name): 
    with open(name, encoding='utf-8', errors='ignore') as f:
        params = {}
        for line in f: 
            if line.startswith('#'):
                line = line.strip('\n')
                line = line.replace('\t', '')
                par = line.split('=')
                val = par[1]
                par = par[0].split('#')[1]
                params[par] = val
                
    return params

def linear(x, m, b): 
    x = np.array(x)
    return m*x + b 


def plot_raman(names, lower, upper, center, fit=fit_lor_exp, sep=0.7, average=False):
    cols = plt.cm.rainbow(np.linspace(0,1,len(names)))
    f, (ax) = plt.subplots(1,1, figsize=(5,5))
    
    if fit == fit_lor_exp: 
        fitted_peak = []
        fitted_peak_err = []
    if fit == fit_lor_lor_exp: 
        fitted_peak1 = []
        fitted_peak1_err = []
        fitted_peak2 = []
        fitted_peak2_err = []
        
    power = []
    yvals = []
        
    for i, name in enumerate(names): 
        if len(name) == 1: 
            name = names
        params = get_params(name)
        
        perc = params['ND Filter']
        lkey = [key for key in params.keys() if 'Laser' in key][0]
        laser = float(params[lkey].split(' ')[0])
        
        temp = pd.read_csv(name, encoding='unicode_escape', header=44, delimiter='\t', names=['cm-1', 'counts'])
        x = np.array(temp['cm-1'])
        y = np.array(temp['counts'])
        if i == 0 and average: 
            av_y = np.zeros_like(y)
        elif i != 0 and average: 
            av_y += y 
            
        wv_min = find_wv(lower, x)
        wv_max = find_wv(upper, x)
        peak_x = x[wv_min:wv_max]
        peak_y = y[wv_min:wv_max]
        
        ax.plot(x, y/peak_y.max() + sep*i, label=perc, color=cols[i])
        ax.set_xlabel('cm-1')
        ax.set_ylabel('counts')
        
        #Get power 
        perc = float(perc.split('%')[0])
        if laser == 532.0: 
            pwr = perc_conv532[perc]*1/100*maxpower532
            power.append(pwr)
        elif laser == 633.0:
            pwr = perc_conv633[perc]*1/100*maxpower633
            power.append(pwr)
            
        else: 
            print(laser)
        
        ax.set_xlabel('cm$^{-1}$')
        ax.set_ylabel('normalized intensity (arb)')
        
        #fit 
        if fit == fit_lor_exp: 
            peak_fit = fit_lor_exp(peak_x, peak_y, peak=center) 
            
            ax.plot(peak_x, peak_fit.best_fit/peak_fit.best_fit.max() + sep*i, color=cols[i])
            peak = round(peak_fit.params['lor_center'].value, 3)
            err = peak_fit.params['lor_center'].stderr
        
            fitted_peak.append(peak)
            if (err) != None: 
                err = round(err, 3)
                fitted_peak_err.append(err)
            else: 
                fitted_peak_err.append(0)
            
            info =[ax, power, fitted_peak, fitted_peak_err]
              
        elif fit == fit_lor_lor_exp: 
            peak_fit = fit_lor_lor_exp(peak_x, peak_y, peak1=center[0], peak2=center[1]) 
            
            ax.plot(peak_x, peak_fit.best_fit/peak_fit.best_fit.max() + sep*i, color=cols[i])
            peak1 = round(peak_fit.params['lor1_center'].value, 3)
            err1 = peak_fit.params['lor1_center'].stderr
            
            peak2 = round(peak_fit.params['lor2_center'].value, 3)
            err2 = peak_fit.params['lor2_center'].stderr
        
            fitted_peak1.append(peak1)
            fitted_peak2.append(peak2)
            if (err1) != None: 
                err1 = round(err1, 3)
                fitted_peak1_err.append(err1)
            else: 
                fitted_peak1_err.append(0)  
                
            if (err2) != None: 
                err2 = round(err2, 3)
                fitted_peak2_err.append(err2)
            else: 
                fitted_peak2_err.append(0) 
                
            
            info = [ax, power, fitted_peak1, fitted_peak1_err, fitted_peak2, fitted_peak2_err]
        elif fit == None: 
            info = [ax, power, x, y] 
            break
        
            
    if average: 
        av_y = av_y/i
        f, (ax2) = plt.subplots(1,1, figsize=(5,5))
        
        if fit == fit_lor_exp: 
            peak_fit = fit_lor_exp(peak_x, peak_y, peak=center) 

            peak = round(peak_fit.params['lor_center'].value, 3)
            err = peak_fit.params['lor_center'].stderr
            
            if err != None: 
                err = round(err, 3)
            else: 
                err = 0
            av_info = [ax2, x, av_y, peak, err]
            ax2.set_title(f'av. of {i+1} scans, peak = {peak}')

        elif fit == fit_lor_lor_exp: 
            peak_fit = fit_lor_lor_exp(peak_x, peak_y, peak1=center[0], peak2=center[1]) 

            peak1 = round(peak_fit.params['lor1_center'].value, 3)
            err1 = peak_fit.params['lor1_center'].stderr

            peak2 = round(peak_fit.params['lor2_center'].value, 3)
            err2 = peak_fit.params['lor2_center'].stderr

            if (err1) != None: 
                err1 = round(err1, 3)
            else: 
                err1 = 0 
            if (err2) != None: 
                err2 = round(err2, 3)
            else: 
                err2 = 0 
            
            av_info = [ax2, av_x, av_y, peak1, err1, peak2, err2]
            ax2.set_title(f'av. of {i+1} scans, peaks = {peak1}, {peak2}')
        
        ax2.scatter(x, av_y, color='k', label='average', s=2)
        ax2.plot(peak_x, peak_fit.best_fit, color='k')
        ax2.set_xlabel('nm')
        ax2.set_ylabel('counts [arb]')
        
        info.append(av_info)
            
    return info     
