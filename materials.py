# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:56:48 2010

@author: dave

"""

import numpy as np
import scipy
import scipy.linalg

# numpy print options:
np.set_printoptions(precision=10, suppress=True)

precision = np.float64

def num(number):
    """
    Force the globally defined precision to a given number
    A string is a valid input, but it seems to have the same results as just
    entering a number.
    """
    if precision.__name__ == 'float128':
        number = np.float128(number)
    elif precision.__name__ == 'float64':
        number = np.float64(number)
    elif precision.__name__ == 'float32':
        number = np.float32(number)
    return number

class Materials:
    """Material properties
    """

    def __init__(self):
        """Material properties
        Class which creates variables with material properties
        The material array has the following lay-out
             0. thickness
             1. E_1
             2. E_2
             3. G_12
             4. nu_12
             5. sig_hat_1t
             6. sig_hat_1c
             7. sig_hat_2t
             8. sig_hat_2c
             9. tau_hat_12
            10. eps_hat_1t
            11. eps_hat_1c
            12. eps_hat_2t
            13. eps_hat_2c
            14. gam_hat_12
            15. density
            16. nu_21
        """
        # number of material properties
        self.nr_prop = 17
        self.fibres()
        self.cores()

        # data structure, same as table 4.2:
        self.struct = ['thickness','E_1','E_2','G_12','nu_12','sig_hat_1t',\
        'sig_hat_1c','sig_hat_2t','sig_hat_2c','tau_hat_12','eps_hat_1t',\
        'eps_hat_1c','eps_hat_2t','eps_hat_2c','gam_hat_12','rho', 'nu_21']


    def cores(self):
        """Materials suited for sandwich cores
        """
        self.cores = dict()

        name = 'PVC_H45'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(48) # density
        self.cores[name][6] = num(-0.55e6) # compressive strength **
        self.cores[name][1] = num(40e6) # compressive modules, E1
        self.cores[name][2] = num(40e6) # compressive modules, E2
        self.cores[name][5] = num(1.2e6)  # tensile strength **
        self.cores[name][7] = num(1.1e6)  # tensile strength ***
        self.cores[name][9] = num(0.5e6)  # shear strength ***
        self.cores[name][3] = num(18e6) # shear modules, G12
        self.cores[name][14] = num(0.1) # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]
        # see page 8.16 of An introduction to Sandwich Structures
        # nu_21 = nu_12 * E_2 / E_1
        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]


        name = 'PVC_H60'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(60) # density
        self.cores[name][6] = num(-0.8e6) # compressive strength **
        self.cores[name][1] = num(60e6) # compressive modules, E1
        self.cores[name][2] = num(60e6) # compressive modules, E2
        self.cores[name][5] = num(1.6e6)  # tensile strength **
        self.cores[name][7] = num(1.4e6)  # tensile strength ***
        self.cores[name][9] = num(0.7e6)  # shear strength ***
        self.cores[name][3] = num(22e6)   # shear modules ***, G12
        self.cores[name][14] = num(0.13)  # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        # eps_hat = sig_hat / E
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]

        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]

        name = 'PVC_H80'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(80) # density
        self.cores[name][6] = num(-1.2e6) # compressive strength **
        self.cores[name][1] = num(85e6) # compressive modules, E1
        self.cores[name][2] = num(85e6) # compressive modules, E2
        self.cores[name][5] = num(2.2e6)  # tensile strength **
        self.cores[name][7] = num(2.0e6)  # tensile strength ***
        self.cores[name][9] = num(1.0e6)  # shear strength ***
        self.cores[name][3] = num(31e6)   # shear modules ***, G12
        self.cores[name][14] = num(0.20)  # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        # eps_hat = sig_hat / E
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]

        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]

        name = 'PVC_H100'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(100) # density
        self.cores[name][6] = num(-1.7e6) # compressive strength **
        self.cores[name][1] = num(125e6) # compressive modules, E1
        self.cores[name][2] = num(125e6) # compressive modules, E2
        self.cores[name][5] = num(3.1e6)  # tensile strength **
        self.cores[name][7] = num(2.4e6)  # tensile strength ***
        self.cores[name][9] = num(1.4e6)  # shear strength ***
        self.cores[name][3] = num(40e6)   # shear modules ***, G12
        self.cores[name][14] = num(0.24)  # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        # eps_hat = sig_hat / E
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]

        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]

        name = 'PVC_H130'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(130) # density
        self.cores[name][6] = num(-2.6e6) # compressive strength **
        self.cores[name][1] = num(175e6) # compressive modules, E1
        self.cores[name][2] = num(175e6) # compressive modules, E2
        self.cores[name][5] = num(4.2e6)  # tensile strength **
        self.cores[name][7] = num(3.0e6)  # tensile strength ***
        self.cores[name][9] = num(2.0e6)  # shear strength ***
        self.cores[name][3] = num(55e6)   # shear modules ***, G12
        self.cores[name][14] = num(0.29)  # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        # eps_hat = sig_hat / E
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]

        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]

        name = 'PVC_H160'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(160) # density
        self.cores[name][6] = num(-3.4e6) # compressive strength **
        self.cores[name][1] = num(230e6) # compressive modules, E1
        self.cores[name][2] = num(230e6) # compressive modules, E2
        self.cores[name][5] = num(5.4e6)  # tensile strength **
        self.cores[name][7] = num(3.9e6)  # tensile strength ***
        self.cores[name][9] = num(2.6e6)  # shear strength ***
        self.cores[name][3] = num(73e6)   # shear modules ***, G12
        self.cores[name][14] = num(0.30)  # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        # eps_hat = sig_hat / E
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]

        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]

        name = 'PVC_H200'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(200) # density
        self.cores[name][6] = num(-4.5e6) # compressive strength **
        self.cores[name][1] = num(310e6) # compressive modules, E1
        self.cores[name][2] = num(310e6) # compressive modules, E2
        self.cores[name][5] = num(7.0e6)  # tensile strength **
        self.cores[name][7] = num(4.8e6)  # tensile strength ***
        self.cores[name][9] = num(3.3e6)  # shear strength ***
        self.cores[name][3] = num(90e6)   # shear modules ***, G12
        self.cores[name][14] = num(0.30)  # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        # eps_hat = sig_hat / E
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]

        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]

        name = 'PVC_H250'
        self.cores[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.cores[name][15] = num(250) # density
        self.cores[name][6] = num(-5.8e6) # compressive strength **
        self.cores[name][1] = num(400e6) # compressive modules, E1
        self.cores[name][2] = num(400e6) # compressive modules, E2
        self.cores[name][5] = num(8.8e6)  # tensile strength **
        self.cores[name][7] = num(6.4e6)  # tensile strength ***
        self.cores[name][9] = num(4.5e6)  # shear strength ***
        self.cores[name][3] = num(108e6)   # shear modules ***, G12
        self.cores[name][14] = num(0.30)  # shear strain ***

        self.cores[name][0] = num(1) # standard thickness, set to 1
        self.cores[name][4] = num(0) # poisson ratio

        self.cores[name][8]  = -self.cores[name][7]
        # eps_hat = sig_hat / E
        self.cores[name][10] = self.cores[name][5] / self.cores[name][1]
        self.cores[name][11] = self.cores[name][6] / self.cores[name][1]
        self.cores[name][12] = self.cores[name][7] / self.cores[name][2]
        self.cores[name][13] = self.cores[name][8] / self.cores[name][2]

        self.cores[name][16] = self.cores[name][4] * self.cores[name][2] / \
                        self.cores[name][1]

        name = 'ZEROS'
        self.cores[name] = scipy.ones((self.nr_prop),dtype=precision)
        self.cores[name] = self.cores[name]*num(1e-20)
        # and set high failure criteria for sigma, tau hats
        self.cores[name][5:10] = num(1e10)
        # and set high failure criteria for epsilon, gamma hats
        self.cores[name][10:15] = num(1)


    def fibres(self):
        """based on:
        Laminate and Sanwich Structures, Foundations of Fibre composites
        table 4.2, page 4.5
        """

        self.fibres = dict()

        name = 'carbon'
        self.fibres[name] = scipy.zeros((self.nr_prop),dtype=precision)
#        self.carbon['thickness']
        self.fibres[name][0] = num(0.000127)
#        self.carbon['E_1']
        self.fibres[name][1] = num(147e9)
#        self.carbon['E_2']
        self.fibres[name][2] = num(9e9)
#        self.carbon['G_12']
        self.fibres[name][3] = num(3.3e9)
#        self.carbon['nu_12']
        self.fibres[name][4] = num(0.31)
#        self.carbon['sig_hat_1t']
        self.fibres[name][5] = num(2260e6)
#        self.carbon['sig_hat_1c']
        self.fibres[name][6] = num(-1200e6)
#        self.carbon['sig_hat_2t']
        self.fibres[name][7] = num(50e6)
#        self.carbon['sig_hat_2c']
        self.fibres[name][8] = num(-190e6)
#        self.carbon['tau_hat_12']
        self.fibres[name][9] = num(100e6)
#        self.carbon['eps_hat_1t']
        self.fibres[name][10] = num(0.015)
#        self.carbon['eps_hat_1c']
        self.fibres[name][11] = num(-0.008)
#        self.carbon['eps_hat_2t']
        self.fibres[name][12] = num(0.005)
#        self.carbon['eps_hat_2c']
        self.fibres[name][13] = num(-0.021)
#        self.carbon['gam_hat_12']
        self.fibres[name][14] = num(0.03)
        self.fibres[name][15] = num(1600)
        self.fibres[name][16] = self.fibres[name][4] * \
            self.fibres[name][2] / self.fibres[name][1]

        name = 'carbon_hm'
        self.fibres[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.fibres[name][0] = num(0.000127)
        self.fibres[name][1] =  num(181e9)
        self.fibres[name][2] =  num(10.3e9)
        self.fibres[name][3] = num(7.17e9)
        self.fibres[name][4] =  num(0.28)
        self.fibres[name][5] = num(1500e6)
        self.fibres[name][6] = num(-1500e6)
        self.fibres[name][7] = num(40e6)
        self.fibres[name][8] = num(-246e6)
        self.fibres[name][9] = num(68e6)
        self.fibres[name][10] = num(0.0083)
        self.fibres[name][11] = num(-0.0083)
        self.fibres[name][12] = num(0.0039)
        self.fibres[name][13] = num(-0.0239)
        self.fibres[name][14] = num(0.0095)
        self.fibres[name][15] = num(1600)
        self.fibres[name][16] = self.fibres[name][4] * \
            self.fibres[name][2] / self.fibres[name][1]

        name = 'e_glass'
        self.fibres[name] = scipy.zeros((self.nr_prop),dtype=precision)
        self.fibres[name][0] = num(0.000127)
        self.fibres[name][1] = num(40e9)
        self.fibres[name][2] = num(9.8e9)
        self.fibres[name][3] = num(2.8e9)
        self.fibres[name][4] = num(0.3)
        self.fibres[name][5] = num(1100e6)
        self.fibres[name][6] = num(-600e6)
        self.fibres[name][7] = num(20e6)
        self.fibres[name][8] = num(-140e6)
        self.fibres[name][9] = num(70e6)
        self.fibres[name][10] = num(0.028)
        self.fibres[name][11] = num(-0.015)
        self.fibres[name][12] = num(0.002)
        self.fibres[name][13] = num(-0.014)
        self.fibres[name][14] = num(0.025)
        self.fibres[name][15] = num(1700)
        self.fibres[name][16] = self.fibres[name][4] * \
            self.fibres[name][2] / self.fibres[name][1]

    def display(self, material, prec_format1=' 1.06f', prec_format2=' 1.02e'):
        """Print in human readable format the material properties
        INPUT: material, the material array
        """

        print ' 0. thickness  :', format(float(material[0]), prec_format1)
        print ' 1. E_1        :', format(float(material[1]), prec_format2)
        print ' 2. E_2        :', format(float(material[2]), prec_format2)
        print ' 3. G_12       :', format(float(material[3]), prec_format2)
        print ' 4. nu_12      :', format(float(material[4]), prec_format1)
        # stressess
        print ' 5. sig_hat_1t :', format(float(material[5]), prec_format2)
        print ' 6. sig_hat_1c :', format(float(material[6]), prec_format2)
        print ' 7. sig_hat_2t :', format(float(material[7]), prec_format2)
        print ' 8. sig_hat_2c :', format(float(material[8]), prec_format2)
        print ' 9. tau_hat_12 :', format(float(material[9]), prec_format2)
        # strains
        print '10. eps_hat_1t :', format(float(material[10]), prec_format1)
        print '11. eps_hat_1c :', format(float(material[11]), prec_format1)
        print '12. eps_hat_2t :', format(float(material[12]), prec_format1)
        print '13. eps_hat_2c :', format(float(material[13]), prec_format1)
        print '14. gam_hat_12 :', format(float(material[14]), prec_format1)
        print '15. rho        :', format(float(material[15]), prec_format2)
        print '16. nu_21      :', format(float(material[16]), prec_format2)
        print ''
