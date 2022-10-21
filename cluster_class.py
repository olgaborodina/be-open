#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

#database handling
import pandas as pd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#scientific calc
import astropy
from astropy.io.votable import parse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

import scipy 
from scipy.optimize import minimize

import sympy

from RegscorePy import aic, bic


class cluster():
    
    def __init__(self, cluster_name):
        self.cluster_name = cluster_name
        
        VizOC_table2 = Vizier(catalog="J/A+A/659/A59/table2")
        VizOC_table1 = Vizier(catalog="J/A+A/659/A59/table1")
        VizOC_table2.ROW_LIMIT = -1
        
        OC_df = VizOC_table2.query_constraints(Cluster=self.cluster_name).values()[0].to_pandas()
        OC_tb1_df = VizOC_table1.query_constraints(Cluster=self.cluster_name).values()[0].to_pandas()
        
        self.sources = OC_df
        self.param = OC_tb1_df
     
    
    
    def plot_scatter(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].scatter(self.sources['RA_ICRS'], self.sources['DE_ICRS'], label=self.cluster_name)
        ax[1].scatter(self.sources['pmRA'], self.sources['pmDE'], label=self.cluster_name)
        
        ax[0].set_title('Ra vs Dec')
        ax[1].set_title('pmra vs pmdec')
        
        ax[0].set_xlabel('RA')
        ax[0].set_ylabel('Dec')
        
        ax[1].set_xlabel('pmra')
        ax[1].set_ylabel('pmdec')
        
        
        
    def distances(self, show_plot=False):
        
        #getting the distance of stars from the center of clusters
        def dist(x1, y1, x2, y2):
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        #getting and converting cluster members' eq coords to cartesian
        eq_coords = SkyCoord(ra=self.sources['RA_ICRS'] * u.degree,
                             dec=self.sources['DE_ICRS'] * u.degree, 
                             distance=(1000 / self.sources['plx']) * u.Parcsec, frame='icrs')
        cart_coords = eq_coords.cartesian.x, eq_coords.cartesian.y, eq_coords.cartesian.z
        
        
        #getting and converting the center coords 
        eq_center_coords = SkyCoord(ra=self.param['RA_ICRS'] * u.degree, 
                                    dec=self.param['DE_ICRS'] * u.degree, 
                                    distance=(1000/self.param['plx']) * u.Parcsec, frame='icrs')
        cart_center_coords = eq_center_coords.cartesian.x, eq_center_coords.cartesian.y, eq_center_coords.cartesian.z
        
        
        d = dist(cart_center_coords[0], cart_center_coords[1],
                 cart_coords[0], cart_coords[1])
        
        self.cart_coords = cart_coords
        self.cart_center_coords = cart_center_coords
        self.center_dist = d
        
        if show_plot:
            #plotting the cluster with radius regions
            plt.figure(figsize=(8, 8))
            plt.scatter(self.cart_coords[0], self.cart_coords[1], label='Data', s=10, alpha=0.5, edgecolors='k')
            plt.scatter(self.cart_center_coords[0], self.cart_center_coords[1], label='Center', marker='x', s=10)

            ax = plt.gca()
            c_max = plt.Circle((self.cart_center_coords[0].value,self.cart_center_coords[1].value), radius=np.max(self.center_dist.value),
                                edgecolor='k',linestyle='-.', fill=False, label='max_dist')
            c_mean = plt.Circle((self.cart_center_coords[0].value,self.cart_center_coords[1].value), radius=np.mean(self.center_dist.value),
                                edgecolor='gray', linestyle='--', fill=False, label='mean_dist')
            ax.add_artist(c_mean)
            ax.add_artist(c_max)
            
            
            plt.xlabel('x [pc]')
            plt.ylabel('y [pc]')
            plt.title('Cartesian Plot')
            plt.legend(loc=[1.01, 0.86])
            plt.grid()
            
    
    def CDF(self):
        
        CDF_x = np.sort(self.center_dist.value)
        x_axis = np.linspace(1e-2, CDF_x.max(), 101)
        CDF = np.arange(1, len(self.center_dist) + 1) / len(self.center_dist)  #normalised CDF
        
        
        # Error propagation
        d_d = lambda plx, eplx: (1000)*((1/plx)**2.*(eplx))
        np.random.seed(None)
        #distance in pc 
        dist = 1000/self.sources['plx']
        #distance errors
        dist_err = d_d(self.sources['plx'], self.sources['e_plx'])
        
        #number of realisations of the CDF
        N=100
        
        CDF_many = np.zeros((len(self.sources['RA_ICRS']),N))
        CDF_x_many = np.zeros((len(self.sources['RA_ICRS']),N))
        
        
        #Setting defaults for centre at function definition
        def make_a_CDF(ra, dec, distance, era, edec, edist, centra, centdec,
                       centdist, ecentdist):
            dist_draw = []
            ra_draw   = []
            dec_draw  = []
            #Uncomment code to include error in cluster centre
            #centra_draw = [] 
            #centdec_draw = []
            centdist_draw = []
            for i in range(len(ra)):
                dist_draw.append(np.random.normal(distance[i], edist[i], size=1)[0])
                ra_draw.append(np.random.normal(ra[i], era[i], size=1)[0])
                dec_draw.append(np.random.normal(dec[i], edec[i], size=1)[0])
                #print(dist_draw, ra_draw, dec_draw)
                #Uncomment to include error in cluster
                #centra_draw.append(np.random.normal(centra, ecentra, size=1)[0])
                #centdec_draw.append(np.random.normal(centdec, ecentdec, size=1)[0])
                #centdist_draw.append(np.random.normal(centdist, ecentdist, size=1)[0])
        #     print(ra_draw)
        
            try1= SkyCoord(ra=ra_draw*u.degree, dec=dec_draw*u.degree, distance=dist_draw*u.Parcsec, frame='icrs')
            try1 = try1.cartesian.x,try1.cartesian.y,try1.cartesian.z
            tryc1= SkyCoord(ra=centra*u.degree, dec=centdec*u.degree, distance=centdist*u.Parcsec, frame='icrs')
            tryc1 = tryc1.cartesian.x,tryc1.cartesian.y,tryc1.cartesian.z
        
        
        
            #getting the distance of stars from the center of clusters
            def dist(x1,y1,x2,y2):
                return ((x2-x1)**2+(y2-y1)**2)**0.5
        
            dtry1 = dist(tryc1[0],tryc1[1],
                     try1[0],try1[1])
        
            CDF_x_try1 = np.sort(dtry1.value)
            CDFtry1 = np.arange(1, len(dtry1) + 1) / len(dtry1)  #normalised CDF
            
            return CDF_x_try1, CDFtry1
        
        
        CDF_err = []
        
        #Loop over the number of realisations we want to create
        for i in range(N):
            CDF_x_many[:,i], CDF_many[:,i] = make_a_CDF(self.sources['RA_ICRS'], self.sources['DE_ICRS'], dist, 
                                              self.sources['e_RA_ICRS'], self.sources['e_DE_ICRS'], dist_err,
                                                       self.param['RA_ICRS'], self.param['DE_ICRS'],
                                                        1000/self.param['plx'],d_d(self.param['plx'], self.param['s_plx']))
            
        #Uncomment to plot realisations.
        #     plt.plot(CDF_x_many[:,i], CDF_many[:,i], alpha=0.05, c="red")
            
        # for each data point compute the standard devation of all the realisations at that value of x
        for k in range(N):
                
            CDF_interp_func = scipy.interpolate.interp1d(CDF_x_many[:,k],CDF_many[:,k], fill_value="extrapolate")
            CDF_err.append(CDF_interp_func(x_axis))
                    
        actual_errors = np.std(np.array(CDF_err), axis=0)

        
        self.CDF = CDF
        self.CDF_x = CDF_x
        self.actual_errors = actual_errors


    def fit_profiles(self, profile_select=['King', 'Plummer', 'Zhao']):
        
        #the profile functions
        def Kings_profile(r, rho_0, r_c, r_t):
            rho_k = rho_0 * (((1 + (r / r_c) ** 2) ** -0.5 - (1 + (r_t / r_c) ** 2) ** -0.5) ** 2)
            mask = r > r_t
            rho_k[mask] = 0
            return rho_k
        
        def Plummers_profile(r, rho_0, a):
            rho_P = rho_0 * ((1 + (r * r / a / a)) ** -2.5)
            return rho_P
        
        def Zhao_profile(r, rho_0, a, beta, gamma):
            rho = rho_0 * ((r / a) ** -gamma) * ((1 + (r / a) ** 2) ** ((gamma - beta) / 2))
            return rho
        
        def dist(x1,y1,x2,y2):
            return ((x2-x1)**2+(y2-y1)**2)**0.5
        
        def pdf2cdf(r, profile, *args):
            
            if profile=='King':
            #getting the cdf from the pdf
                pdf = Kings_profile(r, *args)                 #defining the pdf
                
            elif profile=='Plummer':
                pdf = Plummers_profile(r, *args)                 
        
            elif profile=='Zhao':
                pdf = Zhao_profile(r, *args)             
               
            else:
                print('Please choose a profile between King, Plummer or Zhao')
            
            return np.cumsum(pdf)
        
        def log_likelihood(theta,x,y,y_err,profile):
            model = pdf2cdf(x, profile, *theta)
            sigma2 = y_err ** 2
            return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2)) #neg. chi-sq. + nuisance param
        
        def mle(init_guess,x,y,y_err,profile='King',print_solution=False):
            if profile == 'King':
                bounds = np.array([[0, 0, 10], [1, 10, np.inf]]).T
            elif profile == 'Plummer':
                bounds = np.array([[0, 0], [np.inf, np.inf]]).T
            elif profile == 'Zhao':
                bounds = np.array([[0, 0, 0, 0], [np.inf, np.inf, np.inf, 7]]).T
                                 
            #getting the args for minimise function
            nll = lambda *args: -log_likelihood(*args)
            
            #finding solution by minimise alogorithm
            soln = minimize(nll, init_guess, bounds=bounds
                            , args=(x, y, y_err, profile)) #args = (x,y,yerr,cdf_function)
        
            profiles_dict = {
                'King':['rho_0', 'r_c', 'r_t'],
                'Plummer': ['rho_0', 'a'],
                'Zhao':['rho_0', 'a', 'beta'],
            }
        
            if print_solution==True:
                
                print('-'*30)
                print('Message: ',soln.message)
                print('nhev: ',soln.nfev)
                print('nit: ',soln.nit)
                print('njev: ',soln.njev)
                print('status: ',soln.status)
                print('success: ',soln.success)
                print('-'*30)
        
                print("Maximum likelihood estimates:")  
        
                for i in range(0,len(profiles_dict[profile])):
                    print(f"{profiles_dict[profile][i]} = {soln.x[i]:.3f}")
        
                print('-'*30)
        
            else: 
                pass
        
            return soln
        
        x_axis = np.linspace(1e-2, self.CDF_x.max(), 101)
        CDF_interp_func = scipy.interpolate.interp1d(self.CDF_x,self.CDF, fill_value="extrapolate")
        plt.plot(x_axis, CDF_interp_func(x_axis), label='data')
        
        profiles_dict = {
            'King':[],
            'Plummer': [],
            'Zhao':[],
        }
        
        criterion_dict = {
            'King':[],
            'Plummer': [],
            'Zhao':[],
        }

        for profile in profile_select:
            
            #err array 
            np.random.seed(42)
            yerr  = np.array(self.actual_errors) #np.random.normal(size=len(CDF)) / 1e10
            
            if profile == 'King':
                init = np.array([0.1, np.mean(self.center_dist.value), 5 * np.mean(self.center_dist.value)]) #rho_0, r_c, r_t
            elif profile == 'Plummer':
                init = np.array([0.1, np.mean(self.center_dist.value)]) #rho_0, r_c, r_t
            elif profile == 'Zhao':
                init = np.array([0.1, np.mean(self.center_dist.value), 2, 1]) #rho_0, r_c, r_t
                
            #init_kings = np.array([0.1, np.mean(d.value)]) #rho_0, r_c, r_t
            
            
            
            #finding solution by minimise alogorithm
            soln = mle(init_guess=init,
                        x=x_axis,
                        y=CDF_interp_func(x_axis),
                        y_err = yerr,
                        profile=profile,
                        print_solution=True)
            
            profiles_dict[profile] = list(soln.x)
            
            
            AIC = aic.aic(CDF_interp_func(x_axis), pdf2cdf(x_axis, profile, *soln.x), len(init))
            BIC = bic.bic(CDF_interp_func(x_axis), pdf2cdf(x_axis, profile, *soln.x), len(init))
            
            criterion_dict[profile] = [AIC, BIC]
            
            print(profile, '\n AIC: ' + str(AIC), '\n BIC: ' + str(BIC))
            plt.plot(x_axis, pdf2cdf(x_axis, profile, *soln.x), alpha=0.3, lw=3, label=profile)
            plt.legend(fontsize=14)
            plt.xlabel("x")
            plt.ylabel("y")
            
        self.profiles = profiles_dict
        self.criterion = criterion_dict