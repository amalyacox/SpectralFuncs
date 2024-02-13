

import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread, imshow 
from skimage.morphology import erosion, dilation
import scipy 

class afm_data: 
    def __init__(self, fn, size, plot_raw=False): 
        im = np.array(Image.open(fn))
        self.fn = fn 
        self.im = im 
        self.check_and_fix_partial_scan()
        self.px_y, self.px_x = self.im.shape
        
        if plot_raw: 
            plt.imshow(self.im)
            plt.colorbar()

    def check_and_fix_partial_scan(self): 
        xf_scan = len(set(self.im[0]))
        yf_scan = len(set(self.im[:,0]))

        xb_scan = len(set(self.im[-1]))
        yb_scan = len(set(self.im[:,-1]))
        if xf_scan > 1 and yf_scan > 1 and xb_scan > 1 and yb_scan > 1: 
            print('full scan')
            pass   
        elif  xf_scan == 1 and yf_scan == 1 or xf_scan == 1 and yb_scan == 1: 
            print('no scan? check data', self.fn)
            pass  
        else:
            if xf_scan == 1 or xb_scan ==1: 
                nan_rows = []
                for i, row in enumerate(self.im): 
                    if len(set(row)) > 1: 
                        continue 
                    elif len(set(row)) == 1: 
                        nan_rows.append(i)
                mask = np.setdiff1d(np.arange(self.im.shape[0]), nan_rows)
                self.im = self.im[mask]

            elif yf_scan == 1 or yb_scan == 1: 
                nan_rows = []
                for i, row in enumerate(self.im.T): 
                    if len(set(row)) > 1: 
                        continue 
                    elif len(set(row)) == 1: 
                        nan_rows.append(i)
                mask = np.setdiff1d(np.arange(self.im.shape[1]), nan_rows)
                self.im = self.im[:,mask]
            

    def multi_erosion(self, image, kernel, iterations): 
        for i in range(iterations): 
            image = erosion(image, kernel)
        return image 

    def multi_dilation(self, image, kernel, iterations): 
        for i in range(iterations): 
            image = dilation(image, kernel)
        return image 

    def erode(self, plot=False, change_num=False): 
        # hard coded to always use cross kernel 
        # erode / dilate image eto blur it and help with flattening and region selection
        cross = np.array([[0,1,0], [1,1,1], [0,1,0]])

        ites = [2,4,6,8,10,12,14,16,18,20]


        if change_num: 
            fig, ax = plt.subplots(2, 5, figsize=(17,5))

            for n, ax in enumerate(ax.flatten()): 
                ax.set_title(f'Iterations : {ites[n]}', fontsize=16)
                new_im = self.multi_erosion(self.im, cross, ites[n])
                ax.imshow(new_im)
                ax.axis('off')
            fig.tight_layout
            plt.show()
            num = int(input('choose number of erosions'))
            
        else: 
            num = 20 
        self.im_er = self.multi_erosion(self.im, cross, num)

        if plot: 
            plt.figure()
            plt.imshow(self.im_er)
            plt.colorbar()
            plt.title(f'Eroded image, {num} erosions')

    def fit_background(self, quad=False, plot=False):
        # fit a background and flatten 
        self.erode()
        xx, yy = np.meshgrid(np.arange(self.px_x), np.arange(self.px_y))

        if quad: 
            xy = np.array([xx.ravel(), yy.ravel()]).T
            A = np.c_[np.ones(len(self.im_er.ravel())), xy, np.prod(xy, axis=1), xy**2]
            C,_,_,_ = scipy.linalg.lstsq(A, self.im_er.ravel())
            
            # evaluate it on a grid
            x = xx.flatten()
            y = yy.flatten()
            Z = np.dot(np.c_[np.ones(x.shape), x, y, x*y, x**2, y**2], C).reshape(xx.shape)
        else: 
            A = np.c_[xx.ravel(), yy.ravel(), np.ones_like(self.im_er.ravel())]
            C,_,_,_ = scipy.linalg.lstsq(A, self.im_er.ravel())

            Z = C[0]*xx + C[1]*yy + C[2] 

        if plot: 

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection':'3d'}, figsize=(30,10))

            surf = ax1.plot_surface(xx, yy, self.im_er, linewidth=0, antialiased=False)
            ax1.set_title('Eroded surface using for fit')
            ax2.plot_surface(xx, yy, Z)
            ax2.plot_surface(xx, yy, self.im_er)
            ax2.set_title('Fitting plane')
            ax3.plot_surface(xx, yy, self.im - Z)
            ax3.set_title('Subtracting fitted plane from raw data')
        
        self.flattened = self.im - Z
        self.fitted_bkg =  Z
        plt.figure()
        plt.imshow(self.flattened)
        plt.colorbar()
        plt.title('Flattened image')