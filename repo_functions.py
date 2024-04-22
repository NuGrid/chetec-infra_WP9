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
       for iso in isotopes:
           jiso=isos.index(iso)
           if decayed and typ=="post-sn":
              results[typ][iso]=data["mass_fractions_decayed"][:,jiso]
           else:
              results[typ][iso]=data["mass_fractions"][:,jiso]
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
            dum.append(0.)
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