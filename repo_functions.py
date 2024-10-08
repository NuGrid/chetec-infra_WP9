import os
import re
import numpy as np
import matplotlib.pyplot as plt
#import data_dirlib.patches as patch
import time as t
import more_itertools as mit

import h5py
import csv

import datetime
#from datetime import datetime
import ast


from nugridpy import utils as u
from nugridpy import nugridse as mp



### functions to load data (models and solar abundances):

# function to load Sieverding+2018
def get_profiles(fname,isotopes=["mg24"],presn=True,decayed=False):
    """ extracts mass fraction profiles from hdf5 data file. 
       keywork argument isotopes=["al26"] specifies which isotopes to return.
       Return value is a dictionary containing a "pre-sn" and post-sn" key and the mass fractions for each isotope as well as the mass coordinates.
    """
# Open HDF5 file
    data_file=h5py.File(fname)
    results=dict()
    #initialize output dicts
    results["pre-sn"]=dict()
    results["post-sn"]=dict()
    
    types=["post-sn"]
    if presn:
        types.append("pre-sn")
    for typ in types:
       data=data_file[typ]
       mr=data["mass_coordinates_sun"]
       results[typ]["mr"]=mr
       # need to decode binary isotope names to get strings
       isos=[ name.decode() for name in data["isotopes"] ]
       #find index jiso for isotope and get zonal mass fractions
       #for iso in isotopes:
       #    jiso=isos.index(iso)
       #    if decayed and typ=="post-sn":
       #       results[typ][iso]=data["mass_fractions_decayed"][:,jiso]
       #    else:
       #       results[typ][iso]=data["mass_fractions"][:,jiso]
       for iso in isotopes:
        try:
            jiso=isos.index(iso)
            if decayed and typ=="post-sn":
                results[typ][iso]=data["mass_fractions_decayed"][:,jiso]
            else:
                results[typ][iso]=data["mass_fractions"][:,jiso]
        except ValueError:
            print('Sie18, missing isotope set to zero: ',iso); results[typ][iso] = np.zeros(len(mr))
                
    return(results)


# function to read Lawson data
def load_lawson22(file_name, num_species):    
    print(file_name)
    #mass_lines = %sx findstr "mass enclosed" {filename}
    # getting mass enclosed and num particle
    # mass_lines = !grep 'mass enclosed' {file_name}
    mass_lines = []
    with open(file_name, "rt") as f:
        for ln, line in enumerate(f):
            if 'mass enclosed' in line:
                mass_lines.append(line)
    mass = [float(row.split()[3]) for row in mass_lines]
    numpart = [int(row.split()[0][1:]) for row in mass_lines]
    number_of_parts = len(numpart) # number of particles (it may change from model to model)
    print('# particles = ',number_of_parts)

    # open and read abundances for all trajectories
    a,x,z,iso = [],[],[],[]
    with open(file_name, "rt") as f:
        i = 0
        while i < number_of_parts:
            f.readline(); f.readline();
            j = 0
            a_i,x_i,z_i,iso_i = [],[],[],[]
            while j < num_species:
                line = f.readline().split()
                a_i.append(int(line[0]))
                z_i.append(int(line[1]))
                x_i.append(float(line[2]))
                iso_i.append(f"{get_el_from_z(line[1])}-{line[0]}")
                j += 1
            a.append(a_i); z.append(z_i); x.append(x_i); iso.append(iso_i)
            i += 1        

    return (mass, numpart, number_of_parts, a, z, x, iso)



# load solar data
def load_solar(file_name_solar='iniab1.4E-02As09.ppn'):
    '''
    Function loading the solar abundances that we need to get all the ratios. More than one solar is needed, for 
    different models. :( If not specified a default file is loaded.
    Parameter :
    file_name : name of the solar abundance file (NuGrid format)
    Output (what SIMPLE needs):
    s_iso_new : name of isotopes (this will not change for different solar files) 
    s_abu     : solar abundances
    '''
    f=open(file_name_solar,'r')
    solar_data = f.readlines()
    f.close()
    
    s_iso = []; z_iso = []; abu_s = []
    for i in solar_data:
        s_iso.append(i[3:9].strip())
        z_iso.append(int(i[0:3]))
        abu_s.append(float(i[10:]))
        
    s_iso = np.array(s_iso); abu_s = np.array(abu_s) 
    a_iso = [i[2:].strip() for i in s_iso]; a_iso[0] = '1'; a_iso = [int(i) for i in a_iso]
    s_iso[0] = 'h   1' # correction for H1 special name
    # here below some simple way to build the isotope name with the same structure of the models.
    iso_new_s = [i[:2].strip().capitalize()+'-'+i[2:].strip() for i in s_iso]

    return (abu_s, iso_new_s)    



#######################################

# utils functions: rearrange data, make plots, etc


# function to calculate all ratios and slopes from abundance profiles 
def give_ratios(abu, iso_up, iso_down, ref_mu_and_epsilon, ref_slope, method = 'dilution'):
    """ function: returns delta*1000, epsilon*1e4 (or mu*1e6), with and without mass correction, and slopes
    Inputs are: abu: stellar abundances
    iso_abu: isotopes names consistent with stellar abundance arrays
    abu_solar: solar abundances
    iso_solar: isotopes names consistent with the solar abundance array
    iso_up: isotopes I want to calculate the delta for
    iso_down: normalization isotope for delta
    ref_mu_and_epsilon: list of reference isotope used to normalize to get epsilon and mu
    ref_slope: reference isotope used to normalize epsilon*1e4 (ds or dsm) to get the slopes
    method: dummy variable not used here, introduced for give_ratios_gm"""
        
    rho = [abu[e_label.index(iso_up[i])]/\
                  abu[e_label.index(iso_down[i])]/\
                  (s_abu[s_iso_new.index(iso_up[i].replace("*", ""))]/\
                   s_abu[s_iso_new.index(iso_down[i].replace("*", ""))])-1.\
                  for i in range(len(iso_up))]
    ind_ = iso_up.index(ref_mu_and_epsilon[0])
    # ds_maria = ds without applying mass correction (factor)
    ds_maria = np.array(rho) / np.array(rho[ind_])
    # mass correction factor, derived using natural logs of atomic masses
    factor = [np.log(iso_masses[name_iso_masses.index(iso_up[i].replace("*", ""))]/\
                   iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))]) /\
         np.log(iso_masses[name_iso_masses.index(ref_mu_and_epsilon[i].replace("*", ""))]/\
                   iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))])\
         for i in range(len(iso_up))] ; factor = np.array(factor)
    # ds is epsilon (/10000) or mu (/1e6), based on kinetic fractionation law - Steele+ 2012
    # Steele+ 2012 only applied for 1 element. Generalize?
    # or, do we need to generalize here, if we plot e,g., mu(Ni) vs mu(Fe)
    ds = [np.array(ds_maria[i]) - factor[i] for i in range(len(iso_up))]
    # and now the slopes... 
    ind_ = iso_up.index(ref_slope)
    # do we need to generalize here, and allow for ds_element1 and ds_element2...
    # corr_factor_mix = ...
    slope = np.array(ds) / np.array(ds[ind_]) # * corr_factor_mix
    # done
    return(rho, slope)


# function to calculate all ratios and slopes from abundance profiles 
# originally developed by "Georgy Makhatadze" <georgy.makhatadze@csfk.org>

def give_ratios_gm(abu, e_label,iso_masses,name_iso_masses,iso_up, iso_down, iso_norm, iso_slope,\
                iso_chem = None, chem_factor = None, solar_ref_for_ratios = 'iniab1.4E-02As09.ppn',\
                abu_convert = True, s_abu_convert = True, approximation_method = 'dilution',\
                precision = 0.01, starting_dilution_factor = 1, dilution_step = 0.1,\
                iteration_method = 'precision', largest_offset = 0.0001):
    """Function: returns isotope ratios in rho-notation and slopes for internally normalised data
    
    Inputs are:
    abu: stellar abundances
    iso_abu: isotopes names consistent with stellar abundance arrays
    abu_solar: solar abundances
    iso_solar: isotopes names consistent with the solar abundance array
    std_ratio: isotope ratios in standards, denominator should have ratio of 1
    std_iso: isotope names for standards
    
    iso_up: numerator isotopes
    iso_down: denominator isotopes
    iso_norm: normalising isotopes
    iso_slope: abscissa isotope used to get the slopes, must be one of iso_up
    iso_chem: isotope used to scale all elements relative to each other, needs to have its chem_factor = 1
    MP: man... this is bad.. what if Si28 is not in e-label?
    chem_factor: chemical fractionation factors,
        each stellar abundance in iso_up is multiplied by number from this array, must be relative to iso_chem
    abu_convert: should we convert abundances from mass to number units
    s_abu_convert: same but for solar
    approximation_method: slope ind_slope method, can be:
        dilution - artificial dilute sample, similar to Simon+09/ApJ, Makhatadze+23/GCA & maybe more refs
        linear - linearisation from Dauphas+04/EPSL
        better_linear - improved linearisation from Dauphas+14/EPSL & Lugaro+23/EPJA, similar to Steele+12/ApJ
        std_linear - better_linear, but uses isotope ratios of certified standards instead of solar abundances,
            still uses solar abundances for the coefficient between elements
    precision: maximum allowed difference from previous iteration in relative units for the dilution method
    starting_dilution_factor: starting dilution_factor
    dilution_step: how much the factor goes down each iteration
    iteration_method: how dilution_factor is chosen
        largest_offset - by setting the largest allowed mass-independent offset from the solar,
            similar to Ek's notebook for Lugaro+23/EPJA
        precision - by iterating until slopes stop changing
        dot_product - WIP, ask Andrés & Georgy
    """
    
    # getting default values for some arguments
    if iso_chem == None: iso_chem = iso_up[0]
    if chem_factor == None: chem_factor = np.array([1.]*len(iso_up))
    
    
    s_abu, s_iso_new = load_solar(file_name_solar = solar_ref_for_ratios)
    #print(solar_ref_for_ratios,s_abu[s_iso_new.index('O-16')])

    
    # my understanding is that you always want to convert.... why do we need this then?
    # conversion for stellar and solar abundances is happening separately now
    if abu_convert: 
        # abundances converted from mass to number unit
        abu_num = [abu[e_label.index(iso_up[i])]/\
                   iso_masses[name_iso_masses.index(iso_up[i].replace("*",""))]\
                   for i in range(len(iso_up))]
        abu_num = np.array(abu_num)
        # abundances for iso_chem are fixed and calculated here now
        abu_num_chem = np.array(abu[e_label.index(iso_chem)]/\
                                iso_masses[name_iso_masses.index(iso_chem.replace("*",""))])
    elif not abu_convert:
        abu_num = [abu[e_label.index(iso_up[i])] for i in range(len(iso_up))]
        abu_num_chem = np.array(abu[e_label.index(iso_chem)])
    else:
        print('invalid abu_convert')
    if s_abu_convert: 
        # solar abundances converted from mass to number unit
        s_abu_num = [s_abu[s_iso_new.index(iso_up[i].replace("*", ""))]/\
                     iso_masses[name_iso_masses.index(iso_up[i].replace("*",""))]\
                     for i in range(len(iso_up))]
        s_abu_num = np.array(s_abu_num)
        s_abu_num_chem = s_abu[s_iso_new.index(iso_chem.replace("*",""))]/\
        iso_masses[name_iso_masses.index(iso_chem.replace("*",""))]
    elif not s_abu_convert:
        s_abu_num = [s_abu[s_iso_new.index(iso_up[i].replace("*", ""))]\
                     for i in range(len(iso_up))]
        s_abu_num = np.array(s_abu_num)
        s_abu_num_chem = s_abu[s_iso_new.index(iso_chem.replace("*",""))]
    else:
        print('invalid s_abu_convert')
    
    # rho for the stellar source, as defined in Dauphas+04
    if approximation_method == 'std_linear':
        # as deviation from the standard
        # here rho is already masked on iso_up
        rho = [abu_num[i]/\
               abu_num[iso_up.index(iso_down[i])]/\
               (std_ratio[std_iso.index(iso_up[i].replace("*", ""))]/\
                std_ratio[std_iso.index(iso_down[i].replace("*", ""))])-1.\
               for i in range(len(iso_up))]
        rho = np.array(rho)    
    else:    
        # as deviation from the solar
        rho = [abu_num[i]/\
               abu_num[iso_up.index(iso_down[i])]/\
               (s_abu_num[i]/\
                s_abu_num[iso_up.index(iso_down[i])])-1.\
               for i in range(len(iso_up))]
        rho = np.array(rho)
    
    # index for slope isotope
    ind_slope = iso_up.index(iso_slope)
    
    if approximation_method != 'dilution':
        # rho for normalising ratios, as defined in Dauphas+04
        if approximation_method == 'std_linear':            
            # as deviation from the standard
            rho_norm = [abu_num[iso_up.index(iso_norm[i])]/\
                        abu_num[iso_up.index(iso_down[i])]/\
                        (std_ratio[std_iso.index(iso_norm[i].replace("*", ""))]/\
                         std_ratio[std_iso.index(iso_down[i].replace("*", ""))])-1.\
                        for i in range(len(iso_norm))]
            rho_norm = np.array(rho_norm)            
        else:        
            # as deviation from the solar
            rho_norm = [abu_num[iso_up.index(iso_norm[i])]/\
                        abu_num[iso_up.index(iso_down[i])]/\
                        (s_abu_num[iso_up.index(iso_norm[i])]/\
                         s_abu_num[iso_up.index(iso_down[i])])-1.\
                        for i in range(len(iso_norm))]
            rho_norm = np.array(rho_norm)
        
        # the only difference between linear and better_linear is how atomic masses are treated
        if approximation_method == 'linear':            
            # linear
            mass_diff_coef = [(iso_masses[name_iso_masses.index(iso_up[i].replace("*", ""))]-\
                               iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))])/\
                              (iso_masses[name_iso_masses.index(iso_norm[i].replace("*", ""))]-\
                               iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))])\
                              for i in range(len(iso_up))]            
        else:            
            # logarithmic
            mass_diff_coef = [np.log(iso_masses[name_iso_masses.index(iso_up[i].replace("*", ""))]/\
                                     iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))])/\
                              np.log(iso_masses[name_iso_masses.index(iso_norm[i].replace("*", ""))]/\
                                     iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))])\
                              for i in range(len(iso_up))]            
        mass_diff_coef = np.array(mass_diff_coef)
        
        # coefficient that takes care of different elements plotted together
        diff_ele_coef = [abu_num[iso_up.index(iso_down[i])]/\
                         s_abu_num[iso_up.index(iso_down[i])]\
                         for i in range(len(iso_down))]
        diff_ele_coef = np.array(diff_ele_coef)
        
        # linearised mass-independent rho-values with the applied element coefficient
        rho_mind = [(rho[i] - rho_norm[i]*mass_diff_coef[i])*diff_ele_coef[i]\
                    for i in range(len(iso_up))]
        rho_mind = np.array(rho_mind)
        rho_mind_slope = np.array(rho_mind[ind_slope])
        # those numbers do not have any meaning outside of slope calculation

        # slope on plots where iso_up is ordinate and iso_norm is abscissa
        slope = np.array(rho_mind/rho_mind_slope)
        
    elif approximation_method == 'dilution':        
        # starting dilution_factor
        dilution_factor = starting_dilution_factor
        # counter for the iterations
        counter = 0

        # everything below is in a fake diluted sample
        while True:            
            counter = counter + 1
            print('step', counter, 'dilution_factor =', dilution_factor)
            if counter > 42:
                print('reached', counter, 'iterations, time to stop')
                break
            # you are dead !! dilution_factor constrained by dp
            # from the way it is coded. dilution_factor < 1e-15
            
            # ratios (over iso_chem) in a mixture
            ratio_chem_dilute = [(abu_num[i] * dilution_factor * chem_factor[i] +\
                                  s_abu_num[i] * (1.-dilution_factor))/\
                                 (abu_num_chem * dilution_factor +\
                                  s_abu_num_chem * (1.-dilution_factor))\
                                 for i in range(len(iso_up))]
            ratio_chem_dilute = np.array(ratio_chem_dilute)
            # mixing math fixed
            # the equations taken from Rb-Sr lecture notes by Kostitsyn
            # https://wiki.web.ru/wiki/%D0%93%D0%B5%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9_%D1%84%D0%B0%D0%BA%D1%83%D0%BB%D1%8C%D1%82%D0%B5%D1%82_%D0%9C%D0%93%D0%A3:%D0%93%D0%B5%D0%BE%D1%85%D0%B8%D0%BC%D0%B8%D1%8F_%D0%B8%D0%B7%D0%BE%D1%82%D0%BE%D0%BF%D0%BE%D0%B2_%D0%B8_%D0%B3%D0%B5%D0%BE%D1%85%D1%80%D0%BE%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D1%8F
            
            # isotope ratios in the dilute mixture
            ratio_up = [ratio_chem_dilute[i]/\
                        ratio_chem_dilute[iso_up.index(iso_down[i])]\
                        for i in range(len(iso_up))]
            ratio_up = np.array(ratio_up)
            ratio_norm = [ratio_chem_dilute[iso_up.index(iso_norm[i])]/\
                          ratio_chem_dilute[iso_up.index(iso_down[i])]\
                         for i in range(len(iso_norm))]
            ratio_norm = np.array(ratio_norm)

            # internally normalised mass-independent ratios for the mixtures
            ratio_mind = [ratio_up[i]\
                          *((s_abu_num[iso_up.index(iso_norm[i])]/\
                             s_abu_num[iso_up.index(iso_down[i])])/\
                            ratio_norm[i])**\
                          (np.log(iso_masses[name_iso_masses.index(iso_up[i].replace("*", ""))]/\
                                  iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))])/\
                           np.log(iso_masses[name_iso_masses.index(iso_norm[i].replace("*", ""))]/\
                                  iso_masses[name_iso_masses.index(iso_down[i].replace("*", ""))]))\
                          for i in range(len(iso_up))]
            ratio_mind = np.array(ratio_mind)
            ratio_mind_slope = np.array(ratio_mind[ind_slope])

            # mass-independent rho-values for the mixtures
            rho_mind = [ratio_mind[i]/\
                        (s_abu_num[iso_up.index(iso_up[i])]/\
                         s_abu_num[iso_up.index(iso_down[i])])-1.\
                        for i in range(len(iso_up))]
            rho_mind = np.array(rho_mind)
            rho_mind_slope = np.array(rho_mind[ind_slope])

            # slope on plots where iso_up is ordinate and iso_norm is abscissa
            slope = np.array(rho_mind/rho_mind_slope + 1e-200)
            # forces no division by zero
            
            if iteration_method == 'precision':
                
                if counter == 1:
                    slope_prev = np.array(slope)
                    dilution_factor = dilution_factor*dilution_step
                else:
                    # calculates differences between the current iteration and previous
                    slope_diff = np.abs(np.arctan(slope) - np.arctan(slope_prev))
                    # add inverse slope if too big ?
                    # checks if difference is within the specified precision
                    precision_check = False
                    for i in range(len(slope_diff)):
                        # yep... this would not work with e.g., AGBs. 1 mass coordinate, the surface. 
                        # So, no len() but float here.
                        for j in range(len(slope_diff[i])):   
                            if slope_diff[i][j] > precision:
                                precision_check = True
                    if precision_check:
                        slope_prev = np.array(slope)
                        dilution_factor = dilution_factor*dilution_step
                        if dilution_factor < 1e-12:
                            print('dilution_factor too small, max slope_diff =', np.max(slope_diff),\
                                  '\ntook', counter, 'iterations, final dilution_factor',\
                                  dilution_factor*dilution_step)
                            break
                    else:
                        print('took', counter, 'iterations, final dilution_factor', dilution_factor)
                        break
                        
            elif iteration_method == 'dot_product':
                if counter == 1:
                    rho_mind_prev = np.array(rho_mind)
                    rho_mind_slope_prev = np.array(rho_mind_slope)
                    dilution_factor = dilution_factor*dilution_step
                else:
                    precision_check = False
                    for i in range(len(rho_mind)):
                        for j in range(len(rho_mind[i])):
                            dot_product = rho_mind[i][j]*rho_mind_prev[i][j] +\
                            rho_mind_slope[j]*rho_mind_slope_prev[j]
                            max_dot_product = ((rho_mind[i][j]**2.+rho_mind_slope[j]**2.)**(-2.))*\
                            ((rho_mind_prev[i][j]**2.+rho_mind_slope_prev[j]**2.)**(-2.))
                            if (np.abs(dot_product) < 0.9 * max_dot_product) and (max_dot_product != 0):
                                precision_check = True
                    if precision_check:
                        rho_mind_prev = np.array(rho_mind)
                        rho_mind_slope_prev = np.array(rho_mind_slope)  
                        dilution_factor = dilution_factor*dilution_step
                        if dilution_factor < 1e-100:
                            print('dilution_factor too small\ntook', counter,\
                                  'iterations, final dilution_factor',\
                                  dilution_factor/dilution_step)
                            break
                    else:
                        print('took', counter, 'iterations, final dilution_factor', dilution_factor)
                        break
                    
            elif iteration_method == 'largest_offset':
                
                if counter == 1:
                    dilution_factor = largest_offset / np.max(np.abs(rho_mind))
                elif np.max(np.abs(rho_mind)) > largest_offset:
                    dilution_factor = dilution_factor * largest_offset / np.max(np.abs(rho_mind))
                else:
                    print('took', counter, 'iterations, final dilution_factor', dilution_factor)
                    break
                
            else:
                print('invalid iteration_method')
                
    else:
        print('invalid approximation_method')
        
    # done
    return(rho, slope, rho_mind_slope)


def func_species_deck(string_set,abundance,iso_list_spec,iso_list_master):
    """Documentation: 
    # function will return isotope abundance if present in the list, or zero to be added

    string_set:         identifier of the origin set
    abundance:          abundances to be found
    iso_list_spec:      list of isotopes present in the specific model
    iso_list_master:    this should be iso_list, defined by the user """
    dum = []
    for i in iso_list_master:
        try:
            dum.append(abundance[iso_list_spec.index(i)])
        except ValueError:
            dum.append(np.array(abundance[0])*0.)
            print(string_set,', missing isotope set to zero: ',i)
    
    return(dum)


#!Evelyn new initialization to make Lawson et al. faster
global start_time

z_names = ['Neut', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 
           'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
           'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
           'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 
           'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
           'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 
           'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

def print_time(message,start_time):
    print(f"{message} - {(t.time()-start_time):.2f} s")
    

def get_el_from_z(z):
    '''
    Very simple Vfunction that gives the atomic number AS A STRING when given the element symbol.
    Uses predefined a dictionnary.
    Parameter :
    z : string or number
    For the other way, see get_z_from_el
    '''
    z = int(z)
    
    return (z_names[z])    



# here there is the main loading function
# function loading all the data from the models

def load_stellar_data(data_dir,check_data_file,reqmass,iso_list,e_label):
    if check_data_file == False:
        #############################
        # loading Ritter+18 model
        #############################
        fol2mod = data_dir+'R18/'
        # load instances of models
        # 15Msun
        pt_15 = mp.se(fol2mod,'M15.0Z2.0e-02.Ma.0020601.out.h5',rewrite=True)
        cyc_15 = pt_15.se.cycles[-1]
        #pt_15.se.get('temperature')
        t9_cyc_15  = pt_15.se.get(cyc_15,'temperature')
        mass_15 = pt_15.se.get(cyc_15,'mass')
        # 20Msun
        pt_20 = mp.se(fol2mod,'M20.0Z2.0e-02.Ma.0021101.out.h5',rewrite=True)
        cyc_20 = pt_20.se.cycles[-1]
        #pt_20.se.get('temperature')
        t9_cyc_20  = pt_20.se.get(cyc_20,'temperature')
        mass_20 = pt_20.se.get(cyc_20,'mass')
        # 25Msun
        pt_25 = mp.se(fol2mod,'M25.0Z2.0e-02.Ma.0023601.out.h5',rewrite=True)
        cyc_25 = pt_25.se.cycles[-1]
        #pt_25.se.get('temperature')
        t9_cyc_25  = pt_25.se.get(cyc_25,'temperature')
        mass_25 = pt_25.se.get(cyc_25,'mass')
        #################################################################################
        # loading AGB models
        # test case, M=3Msun, Z=0.03, Battino et al., getting only the last surf file
        ################################################################################
        pt_3 = mp.se(data_dir+'agb_surf_m3z2m3/','96101.surf.h5',rewrite=True)
        ### ### ###
        # loading Pignatari+16 model
        fol2mod = data_dir+'P16/'
        # load instances of models
        # 15Msun
        P16_15 = mp.se(fol2mod,'M15.0',rewrite=True)
        cyc_P16_15 = P16_15.se.cycles[-1]
        #pt_15.se.get('temperature')
        t9_cyc_P16_15  = P16_15.se.get(cyc_P16_15,'temperature')
        mass_P16_15 = P16_15.se.get(cyc_P16_15,'mass')
        # 20Msun
        P16_20 = mp.se(fol2mod,'M20.0',rewrite=True)
        cyc_P16_20 = P16_20.se.cycles[-1]
        #pt_20.se.get('temperature')
        t9_cyc_P16_20  = P16_20.se.get(cyc_P16_20,'temperature')
        mass_P16_20 = P16_20.se.get(cyc_P16_20,'mass')
        # 25Msun
        P16_25 = mp.se(fol2mod,'M25.0',rewrite=True)
        cyc_P16_25 = P16_25.se.cycles[-1]
        #pt_25.se.get('temperature')
        t9_cyc_P16_25  = P16_25.se.get(cyc_P16_25,'temperature')
        mass_P16_25 = P16_25.se.get(cyc_P16_25,'mass')
        ### ### ###
        ################################################
        # Loading Lawson+22 
        #################################################
        dir_law = data_dir+'LAW22/'
        models_list = ['M15s_run15f1_216M1.3bgl_mp.txt','M20s_run20f1_300M1.56jl_mp.txt','M25s_run25f1_280M1.83rrl_mp.txt']
        num_species = 5209
        
        start_time = t.time()
        numpart_all = []; massinc_all = []
        anum_all = []; znum_all = []; x_all = []
        iso_name_all = []; num_of_part_all = []
        for i in models_list:
            mass, numpart, number_of_parts, a, z, x, iso = load_lawson22(dir_law+i,num_species)
            massinc_all.append(mass); num_of_part_all.append(number_of_parts); numpart_all.append(numpart) 
            anum_all.append(a); znum_all.append(z); x_all.append(x); iso_name_all.append(iso)
        
        print_time("done with Lawson",start_time)
        ### ### ### 
        ##########################################
        # dir where Sieverdin models are located
        dir_sie = data_dir+'SIE18/'
        file_sie_all = ["s15_data.hdf5","s20_data.hdf5","s25_data.hdf5"]
        ########################################################################
        ### ### ###
        #########################################
        # Rauscher 2002 
        #############################
        dir_rau = data_dir+'R02/'
        
        #models_rau = ['s15a28c.expl_yield']
        models_rau = ['s15a28c.expl_yield','s20a28n.expl_yield','s25a28d.expl_yield']
        
        start_time = t.time()
        
        rau_mass = []; rau_isos = []; rau_x = []
        for i in range(len(models_rau)):
            filename = dir_rau+models_rau[i]
            print(filename)
            f = open(filename,'r')
            head = f.readline(); isos_dum = head.split()[5:] # getting isotopes, not first header names
            dum_a = [re.findall('\d+', ik)[0] for ik in isos_dum] # getting the A from isotope name
            dum_el = [re.sub(r'[0-9]+', '', ik) for ik in isos_dum] # getting the element name from the isotope name
            dum_new_iso = [dum_el[ik].capitalize()+'-'+dum_a[ik] for ik in range(len(isos_dum))]
            rau_isos.append(dum_new_iso) # isotope name that we can use around, just neutron name is different, but not care
            #
            data = f.readlines()[:-2]                        # getting the all data, excepting the last two lines
                                                  # done reading, just closing the file now
            #
            dum = [float(ii.split()[1])/1.989e+33 for ii in data]; rau_mass.append(dum) # converting in Msun too.
            x_dum = []
            
            
            data = [row.split()[3:] for row in data]
            #
            x_dum = np.transpose(data); x_dum = np.asfarray(x_dum,float)
            print_time("x transpose and done",start_time)
            rau_x.append(x_dum)
        ### ### ### 
        ########################
        # data from LC18
        ##########################
        dir_lc18 = data_dir+'LC18/'
    
        models_lc18 = ['015a000.dif_iso_nod','020a000.dif_iso_nod','025a000.dif_iso_nod']
        
        skip_heavy_ = 43 # usedd to skip final ye and spooky abundances (see below)
        
        start_time = t.time()
        
        lc18_mass_1 = []; lc18_isos = []; lc18_x = []
        for i in range(len(models_lc18)):
            filename = dir_lc18+models_lc18[i]
            print(filename)
            f = open(filename,'r')
            # getting isotopes, not first header names, and final ye and spooky abundances (group of isolated isotopes, 
            # probably sorted with artificial reactions handling mass conservation or sink particles approach)
            head = f.readline(); isos_dum = head.split()[4:-skip_heavy_] 
            # correcting names to get H1 (and the crazy P and A)
            isos_dum[0]=isos_dum[0]+'1'; isos_dum[1]=isos_dum[1]+'1'; isos_dum[6]=isos_dum[6]+'1' 
            dum_a = [re.findall('\d+', ik)[0] for ik in isos_dum] # getting the A from isotope name
            dum_el = [re.sub(r'[0-9]+', '', ik) for ik in isos_dum] # getting the element name from the isotope name
            dum_new_iso = [dum_el[ik].capitalize()+'-'+dum_a[ik] for ik in range(len(isos_dum))]
            lc18_isos.append(dum_new_iso) # isotope name that we can use around, just neutron name is different, but not care
            #
            data = f.readlines()[:-1]             # getting the all data, excepting the last fake line (bounch of zeros)
                                                  # done reading, just closing the file now
            #
            dum = [float(ii.split()[0]) for ii in data]; lc18_mass_1.append(dum) # converting in Msun too.
            x_dum = []
            
            
            data = [row.split()[4:-skip_heavy_] for row in data]
            #
            x_dum = np.transpose(data); x_dum = np.asfarray(x_dum,float)
            print_time("x transpose and done",start_time)
            lc18_x.append(x_dum)
        ### ### ### 
        ####################################
    # alright. The previous cells just need to be run once. Once you got instances for all the models,
    # you are good to go with the analysis. 
    # what of the three models I want for each flavour?
    # this is the deck for the models
    
    
    if reqmass not in [15, 20, 25]:
        print("Error. You can choose only 15, 20, 25 solar mass")
        exit()
    
    if reqmass == 15:
        ind_ = 0
    elif reqmass == 20:
        ind_ = 1
    elif reqmass == 25:
        ind_ = 2
    
    
    if check_data_file == False:
        models = {
            15: {
                "r18_exp": pt_15,
                "r18_cyc": cyc_15,
                "r18_t9_cyc": t9_cyc_15,
                "r18_mass": mass_15,
                "p16_exp": P16_15,
                "p16_cyc": cyc_P16_15,
                "p16_t9_cyc": t9_cyc_P16_15,
                "p16_mass": mass_P16_15
            },
            20: {
                "r18_exp": pt_20,
                "r18_cyc": cyc_20,
                "r18_t9_cyc": t9_cyc_20,
                "r18_mass": mass_20,
                "p16_exp": P16_20,
                "p16_cyc": cyc_P16_20,
                "p16_t9_cyc": t9_cyc_P16_20,
                "p16_mass": mass_P16_20
            },
            25: {
                "r18_exp": pt_25,
                "r18_cyc": cyc_25,
                "r18_t9_cyc": t9_cyc_25,
                "r18_mass": mass_25,
                "p16_exp": P16_25,
                "p16_cyc": cyc_P16_25,
                "p16_t9_cyc": t9_cyc_P16_25,
                "p16_mass": mass_P16_25
            }
        }
    
        # Ritter+18
        r18_exp = models[reqmass]["r18_exp"]
        r18_cyc = models[reqmass]["r18_cyc"]
        r18_t9_cyc = models[reqmass]["r18_t9_cyc"]
        r18_mass = models[reqmass]["r18_mass"]
        
        # Pignatari+16
        p16_exp = models[reqmass]["p16_exp"]
        p16_cyc = models[reqmass]["p16_cyc"]
        p16_t9_cyc = models[reqmass]["p16_t9_cyc"]
        p16_mass = models[reqmass]["p16_mass"]
        
        # lawson+22 # 0=15Msun; 1=20Msun; 2=25Msun
        numpart = numpart_all[ind_]; la22_mass = massinc_all[ind_]
        anum = anum_all[ind_]; znum = znum_all[ind_]; x = x_all[ind_]
        iso_name = iso_name_all[ind_]; num_of_part = num_of_part_all[ind_]
        
        # sieverdin et al # 0=15Msun; 1=20Msun; 2=25Msun
        file_sie = file_sie_all[ind_]
        
        # Rauscher+02 # 0=15Msun; 1=20Msun; 2=25Msun
        rau02_mass = rau_mass[ind_]; rau02_abund_1 = rau_x[ind_]
        rau_isos_1 = rau_isos[ind_]
        
        # Limongi & Chieffi 2018 # 0=15Msun; 1=20Msun; 2=25Msun
        lc18_mass = lc18_mass_1[ind_]; lc18_x_1 = lc18_x[ind_]
        lc18_isos_1 = lc18_isos[ind_]
        
        ################################
        # Battino et al. -test case AGB
        ###############################
        sparsity_surf = 100
        
        pt_agb  = pt_3
        time_ev = pt_3.se.ages[0::sparsity_surf]
        print(len(time_ev))
        ### ### ### 
        ##############################################################################################
        # getting mass and abundances for single isotopes, or single isotopes + radiogenic "by hand", 
        # or elements by adding isotopes    
        ###############################################################################################
        # ritter+18
        cyc_ = r18_cyc 
        r18_mass = r18_exp.se.get(cyc_,'mass')
        
        # not loading the full arrays all the time, just once, and transposing around
        dum_ab = np.transpose(r18_exp.se.get(cyc_,'iso_massf')); dum_iso = r18_exp.se.isotopes
        
        r18_abund = []
        for i in iso_list:
            #iso_abund = 0.; tmp = [r18_exp.se.get(cyc_,'iso_massf',ii) for ii in i]
            iso_abund = 0.; tmp = func_species_deck('rit18',dum_ab,dum_iso,i)
            iso_abund = [np.sum(tmp,axis = 0)][0]
            r18_abund.append(iso_abund)
            
        r18_abund = np.array(r18_abund)
        
        # pignatari+16
        cyc_ = p16_cyc 
        p16_mass = p16_exp.se.get(cyc_,'mass')
        
        # not loading the full arrays all the time, just once, and transposing around
        dum_ab = np.transpose(p16_exp.se.get(cyc_,'iso_massf')); dum_iso = p16_exp.se.isotopes
        
        p16_abund = []
        for i in iso_list:
            #iso_abund = 0.; tmp = [r18_exp.se.get(cyc_,'iso_massf',ii) for ii in i]
            iso_abund = 0.; tmp = func_species_deck('p16',dum_ab,dum_iso,i)
            iso_abund = [np.sum(tmp,axis = 0)][0]
            p16_abund.append(iso_abund)
            
        p16_abund = np.array(p16_abund)
        
        
        # Lawson+22
        la22_abund = []
        for j in iso_list:
            tmp = []
            for jj in j:
                dum = [x[i][iso_name[i].index(jj)] for i in range(num_of_part)]
                tmp.append(dum)
            iso_abund = [np.sum(tmp,axis = 0)][0]
            la22_abund.append(iso_abund)
        
        la22_abund = np.array(la22_abund)
            
        # Sieverdin+
        flat = [x for sublist in iso_list for x in sublist]
        iso_list_sie = [s.replace("-", "").lower() for s in flat]
        #print(iso_list_sie)
        results=get_profiles(dir_sie+file_sie,isotopes=iso_list_sie,decayed=False)    
        
        sie_abund = []
        for i in iso_list:
            iso_abund = 0.; tmp = [results["post-sn"][iso.replace("-", "").lower()] for iso in i]
            iso_abund = [np.sum(tmp,axis = 0)][0]
            sie_abund.append(iso_abund)
        
        sie_abund = np.array(sie_abund)
            
            
        mass_sie = results["post-sn"]["mr"] # mass
        
        
        # Rauscher+2002
        rau02_abund = []
        for j in iso_list:
            #tmp = [rau02_abund_1[rau_isos_1.index(jj)] for jj in j]
            tmp = func_species_deck('rau02',rau02_abund_1,rau_isos_1,j) 
            #print(tmp)
            rau_iso_abund = [np.sum(tmp,axis = 0)][0]
            rau02_abund.append(rau_iso_abund)
        
        rau02_abund = np.array(rau02_abund)
            
        # LC+2018
        lc18_abund = []
        for j in iso_list:
            #tmp = [lc18_x_1[lc18_isos_1.index(jj)] for jj in j]
            tmp = func_species_deck('lc18',lc18_x_1,lc18_isos_1,j)
            #print(tmp)
            lc18_iso_abund = [np.sum(tmp,axis = 0)][0]
            lc18_abund.append(lc18_iso_abund)
        
        lc18_abund = np.array(lc18_abund) 
        ### ### ### 
        ###############################
        # AGB surf test
        # not loading the full arrays all the time, just once, and transposing around
        #dum_ab = np.transpose(pt_3.se.get('iso_massf')[0::sparsity_surf]); dum_iso = pt_3.se.isotopes
        #dum_ab = [pt_3.se.get(int(dum_cyc),'iso_massf') for dum_cyc in pt_3.se.cycles[0::sparsity_surf]]
        #
        # oh no... issue in data written in surf files, some as arrays some as list(array(...)). Talk with Umberto, check
        # mppnp revision (an oldy??) and dealing with allocatable arrays. So, just grabbing the last one, bah... 
        #dum_ab = pt_3.se.get('iso_massf')[-1]
        ccc = pt_agb.se.cycles[-1] ; dum_ab = pt_agb.se.get(ccc,'iso_massf')
        dum_ab = np.array(dum_ab)
        # ! *** !
        # this is done to create from the single surf file a 2d array. This is temporary, since in give_ratio_gm there is a check
        # assuming that there is more than 1 mass coordinate given in dilution/precision setup
        dum_ab = np.stack((dum_ab,dum_ab), axis=0)
        #
        dum_iso = pt_agb.se.isotopes
        
        agb_abund = []
        for i in iso_list:
            #iso_abund = 0.; tmp = [pt_3.se.get(cyc_,'iso_massf',ii) for ii in i]
            iso_abund = 0.; tmp = func_species_deck('bat20',np.transpose(dum_ab),dum_iso,i)
            iso_abund = [np.sum(tmp,axis = 0)][0]
            agb_abund.append(iso_abund)
        agb_abund = np.array(agb_abund)
        ### ### ### 
        #################################################################################
        # preparation for light mode version: write all abundances in an external file:
        # notice the crazy array of lists to list... this is not needed, just to check to be able to write and read
        # multiple times for testing, writing lists as strings and read arrays from strings in the cell below.
        
        data_pot= open('selected_abundances_data.txt','w')
        
        data_pot.write(datetime.date.today().strftime('%Y-%m-%d %H:%M:%S')+' file with selected species - light mode Marco'+'\n')
        # list of species to check and labels
        data_pot.write(str(iso_list)+' \n') # isotope list
        data_pot.write(str(np.array(list(e_label)).tolist())+' \n')  # list for labels in plots
        # rau02
        data_pot.write(str(list(rau02_mass))+' \n')  # list of Rau02 mass coordinates
        data_pot.write(str(np.array(list(rau02_abund)).tolist())+' \n')  # list of Rau02 abundances for given mass coordinates
        # rit18
        data_pot.write(str(np.array(list(r18_mass)).tolist())+' \n')  # list of Rit18 mass coordinates
        data_pot.write(str(np.array(list(r18_abund)).tolist())+' \n')  # list of Rit18 abundances for given mass coordinates
        # let's add some pain here... the temperature array to check the masscut in the processed ejecta
        data_pot.write(str(np.array(list(r18_t9_cyc)).tolist())+' \n')
        # pgn16
        data_pot.write(str(np.array(list(p16_mass)).tolist())+' \n')  # list of Pgn16 mass coordinates
        data_pot.write(str(np.array(list(p16_abund)).tolist())+' \n')  # list of Pgn16 abundances for given mass coordinates
        # let's add some pain here... the temperature coordinate to check the processed ejecta
        data_pot.write(str(np.array(list(p16_t9_cyc)).tolist())+' \n')
        # law22
        data_pot.write(str(np.array(list(la22_mass)).tolist())+' \n')  # list of Law22 mass coordinates
        data_pot.write(str(np.array(list(la22_abund)).tolist())+' \n')  # list of Law22 abundances for given mass coordinates
        # sie18
        data_pot.write(str(np.array(list(mass_sie)).tolist())+' \n')  # list of Sie18 mass coordinates
        data_pot.write(str(np.array(list(sie_abund)).tolist())+' \n')  # list of Sie18 abundances for given mass coordinates
        # LC18
        data_pot.write(str(np.array(list(lc18_mass)).tolist())+' \n')  # list of LC18 mass coordinates
        data_pot.write(str(np.array(list(lc18_abund)).tolist())+' \n')  # list of LC18 abundances for given mass coordinates
        # bat20
        data_pot.write(str(np.array(list(agb_abund)).tolist())+' \n')  # list of Bat20 abundances for the surface (final step)
        
        
        
        data_pot.close()
        print('All models loaded')
        return()
 
   
    
        
        
    


#def func_el_corr(whatever, abu, ref_up, ref_down, mode=0,file_fractionation=None,what_in_file=None):
#    """Documentation: 
#    # mode = 0: c= (ref_EL1_el2/ref_el1_EL2)_sun/(ref_EL1_el2/ref_el1_EL2)_sun = 1. Same without correction factor;
#    # mode = 1: c= (ref_EL1_el2/ref_el1_EL2)_star*/(ref_EL1_el2/ref_el1_EL2)_sun
#    # mode = 2: c= (ref_EL1_el2/ref_el1_EL2)_file/(ref_EL1_el2/ref_el1_EL2)_sun
#    # function will return whatever * correction factor
#
#    whatever:           whatever is read, delta, epsilon, mu, slope...
#    abu:                abundances from the models used to calculate the whatever loaded
#    ref_up:             ref element 1 in normalization
#    ref_down:           ref element 2 in normalization
#    mode:               see above
#    file_fractionation: file from wich it is read the element fractionation in the sample
#    what_in_file:       specify what correction is needed, given the case of interest """
#    
#    if mode == 0:
#        
#        whatever_corrected = whatever
#    
#    elif mode == 1:
#        
#        c = abu[e_label.index(ref_up)]/abu[e_label.index(ref_down)]/\
#             (s_abu[s_iso_new.index(ref_up.replace("*", ""))]/\
#              s_abu[s_iso_new.index(ref_down.replace("*", ""))])
#        whatever_corrected = whatever * np.array(c)
#    
#    elif mode == 2:
#        
#        # open file and read the fractionation you want from what_in_file
#        f_ = open(file_fractionation, 'r')
#        header = f_.readline()
#        if header.split()[0] != ref_up.rpartition('-')[0]:
#            print('Element 1 in '+file_fractionation+' does not match '+ref_up+'! Stop!')
#        if header.split()[1] != ref_down.rpartition('-')[0]:
#            print('Element 2 in '+file_fractionation+' does not match '+ref_down+'! Stop!')
#        for i in f_.readlines():
#            if i.split()[0] == what_in_file:
#                c = float(i.split()[1])
#                break
#        f_.close() # done with the file
#        c = c/(s_abu[s_iso_new.index(ref_up.replace("*", ""))]/s_abu[s_iso_new.index(ref_down.replace("*", ""))])
#        whatever_corrected = whatever * np.array(c)
#    
#    return(whatever_corrected)
#

# utility function to generate automatically CCSN ejecta zone structure in abundance plots (e.g., mass fractions vs mass coordinates). see e.g., Meyer+ 1995
def structure(all_models, reqmass, rmodel, y2, file_name_structure='file_name', write_structure_data=False):

    mass = all_models[rmodel][reqmass]['masscoord']
    he4  = all_models[rmodel][reqmass]['abu']['He-4']
    c12  = all_models[rmodel][reqmass]['abu']['C-12'] 
    ne20 = all_models[rmodel][reqmass]['abu']['Ne-20'] 
    o16  = all_models[rmodel][reqmass]['abu']['O-16'] 
    si28 = all_models[rmodel][reqmass]['abu']['Si-28'] 
    n14  = all_models[rmodel][reqmass]['abu']['N-14'] 
    ni56 = all_models[rmodel][reqmass]['abu']['Ni-56'] 
    masscut = mass[0]
    massmax = mass[-1]
    print("m_cut: "+str(masscut))
    print("massmax: "+str(massmax))

    si_layer_log = False ; ni_layer_log = False # default, switched to true if they are found
    # definition of borders
    ih = np.where((he4 > 0.5))[0][-1]
    print("Lower boundary of the H shell: "+str(mass[ih]))

    ihe1 = np.where((n14 > o16) & (n14 > c12) & (n14 > 1.e-3))[0][0]
    print("Lower boundary of the He/N shell: "+str(mass[ihe1]))

    ihe = np.where((c12 > he4) & (mass <= mass[ih]))[0][-1]
    print("Lower boundary of the He/C shell: "+str(mass[ihe]))

    ic2 = np.where((c12 > ne20)&(si28<c12)&(c12> 8.e-2))[0][0]
    print("Lower boundary of the O/C shell: "+str(mass[ic2]))

    ine = np.where((ne20 > 1.e-3) & (si28 < ne20) & (ne20 > c12))[0][0]
    if ine>ic2:
        ine=ic2
    print("Lower boundary of the O/Ne shell: "+str(mass[ine]))

    io = np.where((si28 < o16) & (o16 > 5.e-3))[0][0]
    print("Lower boundary of the O/Si layer: "+str(mass[io]))

    try:
        isi = np.where((ni56 > si28))[0][-1]
        print("Lower boundary of the Si layer: "+str(mass[isi]))
        si_layer_log = True
    except IndexError:
        isi=0
        print("No lower boundary of Si layer")
    
    try:
        ini = np.where((ni56 > si28) & (mass > masscut))[0][0]
        print("Lower boundary of the Ni layer: "+str(mass[ini]))
        ni_layer_log = True
    except IndexError:
        ini=0.2
        print("No Ni layer")

    # write structure data in a file for use outside simple:
    if write_structure_data:
        file_ = open(file_name_structure,'a')
        file_.write('source: '+rmodel+'\n')
        file_.write('initial mass (Msun): '+str(reqmass)+'\n')
        file_.write('outer mass coordinate:            '+str('%6E' % massmax)+' : # \n')
        file_.write('Lower boundary of the H shell:    '+str('%6E' % mass[ih])+' : H \n')
        file_.write('Lower boundary of the He/N shell: '+str('%6E' % mass[ihe1])+' : He/N \n')
        file_.write('Lower boundary of the He/C shell: '+str('%6E' % mass[ihe])+' : He/C \n')
        file_.write('Lower boundary of the O/C shell:  '+str('%6E' % mass[ic2])+' : O/C \n')
        file_.write('Lower boundary of the O/Ne shell: '+str('%6E' % mass[ine])+' : O/Ne \n')
        file_.write('Lower boundary of the O/Si layer: '+str('%6E' % mass[io])+' : O/Si \n')
        if si_layer_log:
            file_.write('Lower boundary of the Si layer:   '+str('%6E' % mass[isi])+' : Si \n')
        if ni_layer_log:
            file_.write('Lower boundary of the Ni layer:   '+str('%6E' % mass[ini])+' : Ni \n')
        file_.write('masscut:                          '+str('%6E' % masscut)+' : # \n')
        file_.write('############# \n')
        file_.close()
        
    # plot limits
    xl1 = masscut-0.25
    xl2 = np.round(massmax,1)
    xln = np.round(massmax,1)
    
    #remnant
    plt.gca().axvline(x = masscut, color = 'black', linestyle = ':',lw=2, alpha=0.75)
    plt.gca().text((((xl1+masscut)/2)), y2*1.3,r'M$_{\rm rem}$',fontsize=10.,color='black',horizontalalignment='center')
    plt.gca().fill_between([xl1,masscut],[-1.e10,-1.e10],[1.e10,1.e10],color='gray',alpha=0.5)
    
    #Convective envelope
    if mass[ih]+1<massmax:
        xl2=np.round(mass[ih]+0.5,1)
        plt.gca().axvline(x = mass[ih], color = 'black', linestyle = '--',lw=2, alpha=0.75)
        plt.gca().fill_between([mass[ih],xl2],[-1.e10,-1.e10],[1.e10,1.e10],color='lightblue',alpha=0.25)
        if np.round(mass[ih],2)!=np.round(massmax,2):
            plt.gca().text((((mass[ih]+xl2)/2)), y2*1.3,'H',fontsize=10.,color='black',horizontalalignment='center')
            
    #He/N layer
    if mass[ihe]+0.05<massmax:
        plt.gca().axvline(x = mass[ihe], color = 'black', linestyle = '--',lw=2, alpha=0.75)
        if mass[ih]==massmax:
            xl2=np.round(mass[ihe1]+0.3,1)
            plt.gca().text((((xl2+mass[ihe1])/2)), y2*1.3,'He/N',fontsize=10.,color='black',horizontalalignment='center')
        else:
            plt.gca().text((((mass[ih]+mass[ihe1])/2)), y2*1.3,'He/N',fontsize=10.,color='black',horizontalalignment='center')

    #He/C layer
    if mass[ihe1]+0.04<massmax:
        plt.gca().axvline(x = mass[ihe1], color = 'black', linestyle = '--',lw=2, alpha=0.75)
        plt.gca().text((((mass[ihe]+mass[ihe1])/2)), y2*1.3,'He/C',fontsize=10.,color='black',horizontalalignment='center')
        if mass[ih]+0.05>massmax and mass[ihe]+0.05<massmax:
            if ihe-ic2<0.2:
                xl2=np.round(mass[ihe1]+0.5,1)
                plt.gca().text((((int(xl2)+mass[ic2])/2)), y2*1.3,'He/C',fontsize=10.,color='black',horizontalalignment='center')
    
    plt.gca().axvline(x = mass[ic2], color = 'black', linestyle = '--',lw=2, alpha=0.75)
    
    #merger
    #max_value = max([mass[ic2], masscut, mass[ine]])
    if np.abs(mass[ic2]-mass[ine])>0.05:
        plt.gca().axvline(x = mass[ine], color = 'black', linestyle = '--',lw=2, alpha=0.75)
        plt.gca().text((((mass[ic2]+max(mass[ine],masscut))/2)), y2*1.3,'O/Ne',fontsize=10.,color='black',horizontalalignment='center')
        plt.gca().fill_between([max(mass[ine],masscut),mass[ic2]],[-1.e10,-1.e10],[1.e10,1.e10],color='lightblue',alpha=0.25)
    
    plt.gca().axvline(x = mass[io], color = 'black', linestyle = '--',lw=2, alpha=0.75)
    plt.gca().axvline(x = mass[isi], color = 'black', linestyle = '--',lw=2, alpha=0.75)
    
    plt.gca().fill_between([mass[ihe],mass[ihe1]],[-1.e10,-1.e10],[1.e10,1.e10],color='lightblue',alpha=0.25)

    
    plt.gca().text((((max(mass[ic2],masscut)+mass[ihe])/2)), y2*1.3,'O/C',fontsize=10.,color='black',horizontalalignment='center')
        
    if(mass[io]>masscut and mass[io]-masscut>0.05):
        plt.gca().text((((mass[ine]+mass[io])/2)), y2*1.3,'O/Si',fontsize=10.,color='black',horizontalalignment='center')
    elif (mass[ine]-masscut>0.05):
        plt.gca().text((((mass[ine]+masscut)/2)), y2*1.3,'O/Si',fontsize=10.,color='black',horizontalalignment='center')
    
    #Si layer
    if(mass[io]>masscut):
        if(mass[isi]>masscut):
            plt.gca().text((((mass[io]+mass[isi])/2)), y2*1.3,'Si',fontsize=10.,color='black',horizontalalignment='center')
            plt.gca().fill_between([mass[isi],mass[io]],[-1.e10,-1.e10],[1.e10,1.e10],color='lightblue',alpha=0.25) #silicon
        elif (mass[io]-masscut>0.05):
            plt.gca().text((((mass[io]+masscut)/2)), y2*1.3,'Si',fontsize=10.,color='black',horizontalalignment='center')
            plt.gca().fill_between([mass[io],masscut],[-1.e10,-1.e10],[1.e10,1.e10],color='lightblue',alpha=0.25) #silicon
    
    #Ni layer
    if(mass[isi]>masscut):
        plt.gca().text((((mass[ini]+mass[isi])/2)), y2*1.3,'Ni',fontsize=10.,color='black',horizontalalignment='center')

    if(xl2-xln>0.3):
        xl2 = np.round(massmax,1)+0.1
        
        
    return(xl1,xl2)        


#
# function generating a figure with abundance (in mass fractions) of selected isotopes vs mass fractions, using the function structure.
#
def abundanceplot(rmodel, iso_selected, reqmass, all_models, title_abundance_plot, col, lin, x1=1, x2=10, y1=1e-10, y2=1, \
                  xlabel="Mass coordinate M$_{\odot}$", ylabel="Mass fraction", legendpos='outside',\
                  grid=0, abu_name='nosave', file_name_structure='name_file', onion=True, write_structure_data=True, show_plot=True):
    

    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.ticker import MultipleLocator
    from matplotlib.ticker import LogLocator

    e_label_split = [iso.split("-") for iso in iso_selected]
    title = title_abundance_plot
    
    ifig=plt.figure(figsize=(9,6))

    for i in range(len(iso_selected)):
        label_text = rmodel+" "+str(reqmass)+" "+ str(e_label_split[i][0])+str(e_label_split[i][1])
        plt.semilogy(all_models[rmodel][reqmass]['masscoord'],all_models[rmodel][reqmass]['abu'][iso_selected[i]],marker=lin,\
                     color=col[i], markersize=0.5, ls='-', label=label_text)        

    plt.grid(alpha=grid)
    plt.xlabel(xlabel,fontweight='bold', fontsize=17)
    plt.ylabel(ylabel,fontweight='bold', fontsize=17)
    plt.tick_params(left=True,right=True,top=True,labelleft=True,which='both')
    plt.title(title,fontweight='bold', fontsize=17, y=1.05)

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_major_locator(LogLocator(subs=(1.0, ),numticks=999))
    plt.gca().yaxis.set_minor_locator(LogLocator(numticks=999, subs='auto'))
    
    if onion==True:
        xl1,xl2 = structure(all_models, reqmass, rmodel, y2, file_name_structure=file_name_structure, write_structure_data=True)
        plt.xlim(xl1,xl2)
    else:
        plt.xlim(x1,x2)
    
    plt.ylim(y1,y2)

    if legendpos=='outside':
        plt.legend(bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0)
    else:
        plt.legend()

    if abu_name != 'nosave':
        plt.savefig(abu_name, bbox_inches='tight', pad_inches=0.1,dpi=300)

    if show_plot:
        plt.show(); 
    else: 
        plt.close(ifig)
    return()



#
# function generating a figure with slopes of selected ratios, using the function structure.
#
def ratioplot(rmodel, iso_selected, reqmass, all_models, ylabel, label_for_legend, title, col, lin, x1=1 ,x2=10 ,y1=-15 ,y2=25, xlabel="Mass coordinate M$_{\odot}$",legendpos="inside",\
              grid=0,ratio_name="nosave", onion=True):
    
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.ticker import MultipleLocator
    from matplotlib.ticker import LogLocator

    ifig2=plt.figure(figsize=(10,6))
        
    for i in range(len(iso_selected)):
        label_text = rmodel+" "+str(reqmass)+" "+ str(label_for_legend[i])
        plt.plot(all_models[rmodel][reqmass]['masscoord'],all_models[rmodel][reqmass]['slope'][iso_selected[i]], color=col[i], marker=lin,\
                 ls='-', markersize=0.8, alpha=0.5, label=label_text)
            
    #plt.hlines(3.003,x1,x2)
    #x = np.linspace(x1, x2, 100)
    #plt.fill_between(x, 3.003-0.166, 3.003+0.166, color='lightblue', alpha=0.5)
    
    plt.grid(alpha=grid)
    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel(ylabel, fontsize=17)
    plt.ylim(y1,y2)
    plt.tick_params(left=True,right=True,top=True,labelleft=True,which='both') #,labelright=True)
    plt.title(title,fontweight='bold', fontsize=17, y=1.05)

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    
    if onion==True:
        file_name_structure = 'not_used'
        xl1,xl2 = structure(all_models, reqmass, rmodel, y2, file_name_structure=file_name_structure, write_structure_data=False)
        plt.xlim(xl1,xl2)
    else:
        plt.xlim(x1,x2)

    if legendpos=='outside':
        plt.legend(bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0)
    else:
        plt.legend()

    if ratio_name != "nosave":
        plt.savefig(ratio_name, bbox_inches='tight', pad_inches=0.1,dpi=300)

    plt.show()