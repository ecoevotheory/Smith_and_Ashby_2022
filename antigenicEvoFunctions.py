'''
Functions for the paper "Antigenic evolution of SARS-CoV-2 in immunocompromised hosts" by C.A. Smith and B. Ashby.
This file contains any functions that enable the simulation of both the between- and within-host models.
It also has functions for the generation of the heat maps

Author: Cameron Smith
Version: 1
Date: 11/01/2022
'''

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% Between-host dynamics

def betweenHost(p = 0.05, xi = 0.8, N = 1e7, n = 30, muH = 0.01, muC = 0.01, R0 = 3, gammaH = 1/7, gammaC = 1/140, tmax = 1460, eta = 5.0, epistasis_type = 'cos'):
    '''
    Function which simulates the evolution of a phenotype with immuno-competent and immuno-suppressed individuals under epistasis
    
    INPUTS:
    -------
    p: double in [0,1] - Proportion who are immuno-suppressed
    xi: double greater than 0 - Strength of epistasis, large values give smaller fitness valleys
    N: int greater than 0 - Number in population
    n: int greater than 0 - Number of phenotypes
    muH: double greater than 0 - Adaption rate for immunocompetent
    muC: double greater than 0 - Adaption rate for immunosupressed
    R0: double greater than 0 - Basic reproductive number
    gammaH: double greater than 0 - Recovery rate of immuno-competent individuals
    gammaC: double greater than 0 - Recovery rate of immuno-suppressed individuals
    tmax: double greater than 0 - Final time (days)
    eta: double greater than 0 - Strength of cross-immunity, large values give low cross immunity
    epistasis_type: string in {'mod', 'quad', 'cos'} - Chooses the type of epistasis valley. 'mod' by default
    
    OUTUTS:
    -------
    Sh_out: ndarray - State through time for suscepible immuno-competent individuls
    Sc_out: ndarray - State through time for suscepible immuno-suppressed individuls
    Ih_out: ndarray - State through time for infected immuno-competent individuls
    Ic_out: ndarray - State through time for infected immuno-suppressed individuls
    params: dict - List of all parameter values inputted into the function
    '''

    # Create a dictionary that contains the parameter values
    params = locals()

    # Phenotype and time vectors
    nVec = np.arange(1, n+1)
    tVec = np.arange(0, tmax)

    # Create a vector which contains the epistasis curve
    if epistasis_type == 'mod':
        epistasis = 2*xi/(n-1)*np.abs(nVec-(n+1)/2)+1-xi  # Absolute value
    elif epistasis_type == 'quad':
        epistasis = 1 + 4*xi/((n-1)**2)*(1-nVec)*(n-nVec)  # Quadratic
    else:
        epistasis = xi/2*np.cos((2*np.pi*(nVec-1))/(n-1)) + 1 - xi/2  # Cosine
        if epistasis_type != 'cos':
            print('Invalid epistais type, cosine function used')
    
    # Add the epistasis function to the dictionary
    params['epistasis'] = epistasis
    
    # Number in each subpopulation
    Nh = N*(1-p)
    Nc = N*p

    # Transmission vector which depends on the phenotype through the epistasis
    beta = R0*gammaH*np.ones(n)*epistasis

    # Cross immunity matrix X
    X = np.zeros((n,n))
    for ii in range(n):
        for jj in range(n):
            if eta > 0:
                X[ii,jj] = np.exp(-(ii-jj)**2/(2*eta))
            else:
                X[ii,jj] = 1*(ii==jj)

    # Add the cross-immunity to the dictionary
    params['X'] = X

    # Initial populations
    Sh = (1-p)*np.ones(n)
    Sc = p*np.ones(n)
    Ih = np.zeros(n)
    Ih[0] = 1e-6
    Ic = np.zeros(n)
    Sh = Sh - X[:,1]*Ih[0]

    # Initalise matrices
    Sh_out = np.zeros((n,tmax))
    Sh_out[:,0] = Sh
    Sc_out = np.zeros((n,tmax))
    Sc_out[:,0] = Sc
    Ih_out = np.zeros((n,tmax))
    Ih_out[:,0] = Ih
    Ic_out = np.zeros((n,tmax))
    Ic_out[:,0] = Ic

    # Loop through time and dynamically adjust the populations
    for t in range(1, tmax):

        # Calculate the number of new infections in the two subpopulations. This is done by using a Poisson random variable
        if p == 0:
            Sh_infection = np.minimum(Sh, np.random.poisson(Nh*beta*Sh*(Ih+Ic)).astype('float')/Nh)
        elif p == 1:
            Sc_infection = np.minimum(Sc, np.random.poisson(Nc*beta*Sc*(Ih+Ic)).astype('float')/Nc)
        else:
            Sh_infection = np.minimum(Sh, np.random.poisson(Nh*beta*Sh*(Ih+Ic)).astype('float')/Nh)
            Sc_infection = np.minimum(Sc, np.random.poisson(Nc*beta*Sc*(Ih+Ic)).astype('float')/Nc)

        # Calculate the effect of cross-immunity on these infections
        if p == 0:
            Sh_cross_immunity = np.minimum(Sh, np.sum(np.tile(Sh_infection, (n, 1))*X, axis=1))
        elif p == 1:
            Sc_cross_immunity = np.minimum(Sc, np.sum(np.tile(Sc_infection, (n, 1))*X, axis=1))
        else:
            Sh_cross_immunity = np.minimum(Sh, np.sum(np.tile(Sh_infection, (n, 1))*X, axis=1))
            Sc_cross_immunity = np.minimum(Sc, np.sum(np.tile(Sc_infection, (n, 1))*X, axis=1))

        # Calculate the number of recoveries in each subpopulation using a Poisson random variable
        if p == 0:
            Sh_recovery = np.minimum(Ih, np.random.poisson(Nh*Ih*gammaH).astype('float')/Nh)
        elif p == 1:
            Sc_recovery = np.minimum(Ic, np.random.poisson(Nc*Ic*gammaC).astype('float')/Nc)
        else:
            Sh_recovery = np.minimum(Ih, np.random.poisson(Nh*Ih*gammaH).astype('float')/Nh)
            Sc_recovery = np.minimum(Ic, np.random.poisson(Nc*Ic*gammaC).astype('float')/Nc)   

        # Calculate the number of mutations
        probs = np.hstack((1, 0.5*np.ones(n-2), 0))  # The probabilities that each mutation is an upwards mutation, assuming that upwards and downwards are equally likely
        if p == 0:
            Sh_mutation = np.minimum(Ih, np.random.poisson(Nh*muH*Ih).astype('float')/Nh)
            Sh_right_mut = np.random.binomial((Nh*Sh_mutation).astype('int'), probs).astype('float')
            Sh_left_mut = Sh_mutation - Sh_right_mut
        elif p == 1:
            Sc_mutation = np.minimum(Ic, np.random.poisson(Nc*muC*Ic).astype('float')/Nc)
            Sc_right_mut = np.random.binomial((Nc*Sc_mutation).astype('int'), probs).astype('float')
            Sc_left_mut = Sc_mutation - Sc_right_mut
        else:
            Sh_mutation = np.minimum(Ih, np.random.poisson(Nh*muH*Ih).astype('float')/Nh)
            Sh_right_mut = np.random.binomial((Nh*Sh_mutation).astype('int'), probs).astype('float')
            Sh_left_mut = Sh_mutation - Sh_right_mut
            Sc_mutation = np.minimum(Ic, np.random.poisson(Nc*muC*Ic).astype('float')/Nc)
            Sc_right_mut = np.random.binomial((Nc*Sc_mutation).astype('int'), probs).astype('float')
            Sc_left_mut = Sc_mutation - Sc_right_mut

        # Update numbers in each population
        if p == 0:
            Sh = np.maximum(np.zeros(n), Sh - Sh_cross_immunity)
            Ih = np.maximum(np.zeros(n), Ih + Sh_infection - Sh_recovery - Sh_mutation/Nh + np.hstack((0, Sh_right_mut[0:-1]))/Nh + np.hstack((Sh_left_mut[1:], 0))/Nh)
        elif p == 1:
            Sc = np.maximum(np.zeros(n), Sc - Sc_cross_immunity)
            Ic = np.maximum(np.zeros(n), Ic + Sc_infection - Sc_recovery - Sc_mutation/Nc + np.hstack((0, Sc_right_mut[0:-1]))/Nc + np.hstack((Sc_left_mut[1:], 0))/Nc)
        else:
            Sh = np.maximum(np.zeros(n), Sh - Sh_cross_immunity)
            Ih = np.maximum(np.zeros(n), Ih + Sh_infection - Sh_recovery - Sh_mutation/Nh + np.hstack((0, Sh_right_mut[0:-1]))/Nh + np.hstack((Sh_left_mut[1:], 0))/Nh)
            Sc = np.maximum(np.zeros(n), Sc - Sc_cross_immunity)
            Ic = np.maximum(np.zeros(n), Ic + Sc_infection - Sc_recovery - Sc_mutation/Nc + np.hstack((0, Sc_right_mut[0:-1]))/Nc + np.hstack((Sc_left_mut[1:], 0))/Nc)

        # Update solution matrices
        Sh_out[:,t] = Sh
        Sc_out[:,t] = Sc
        Ih_out[:,t] = Ih
        Ic_out[:,t] = Ic

    return([Sh_out, Sc_out, Ih_out, Ic_out, params])

#%% Code to classify the output of betweenHost according to the heat maps

def classify(betweenHost_output, threshold = 10**(-2)):
    '''
    Code to classify the output of the function betweenHost.

    INPUTS:
    -------
    betweenHost_output: Output from a betweenHost function call to be classified
    threhold: Threshold value below which counts are considered 0

    OUTPUTS:
    --------
    max(L): The maximum distance between observed variants
    numVar: The total number of observed variants
    '''

    # Extrat the omicron2 output
    I = betweenHost_output[2] + betweenHost_output[3]
    n = betweenHost_output[4]['n']
    nt = betweenHost_output[4]['tmax']

    # Convert I by firstly checking which values are over a threshold
    J = I > threshold

    # Number of variants seen
    numVar = np.sum(np.sum(J, axis=1) > 0)

    # Create a vector which will store the first time a variant is seen
    minimums = np.zeros(n)
    for ii in range(n):
        X = np.nonzero(J[ii,])
        if X[0].size == 0:
            minimums[ii] = 0
        else:
            minimums[ii] = np.min(X)

    # Place a 1 where the mutant is seen and 0 otherwise
    Z = minimums > 0

    # Generate a score for this based on the maximum size of the gaps
    L = [0]
    count = 0
    for ii in range(n):
        if Z[ii] == 0:
            count += 1
        else:
            L.append(count)
            count = 0
    
    # Output the maximum gap between variants that appear
    return(max(L), numVar)

#%% Heat map over cross immunity and epistasis strength

def heatMapCrossEpi(dataset = None, filename = None, etaHigh = 10.0, numVec = 21, M = 10, threshold = 10**(-2), p = 0.05, N = 1e7, n = 30, muH = 0.01, muC = 0.01, R0 = 3, gammaH = 1/7, gammaC = 1/140, tmax = 1460, epistasis_type = 'cos'):
    '''
    Code to either plot or generate a heat map over the cross-immunity and epistasis strengths.
    If only a plot is required, specify a dataset which is the output of this function.
    If you need to run dataset, please specify the parameters but leave datasets=None

    INPUTS:
    -------
    dataset: Filename where the output is stored without extension. Should be a .pkl file
    filename: Filename for if a dataset is to be created without extension
    etaHigh: The maximum value of eta. It is assumed the lower bound is 0
    numVec: The number in each parameter vector
    M: Number of independent repeats per data point
    threshold: The threshold for the classify function

    p: double in [0,1] - Proportion who are immuno-suppressed
    N: int greater than 0 - Number in population
    n: int greater than 0 - Number of phenotypes
    muH: double greater than 0 - Adaption rate for immunocompetent
    muC: double greater than 0 - Adaption rate for immunosupressed
    R0: double greater than 0 - Basic reproductive number
    gammaH: double greater than 0 - Recovery rate of immuno-competent individuals
    gammaC: double greater than 0 - Recovery rate of immuno-suppressed individuals
    tmax: double greater than 0 - Final time (days)
    epistasis_type: string in {'mod', 'quad', 'cos'} - Chooses the type of epistasis valley. 'mod' by default
    
    OUTPUTS:
    --------
    datasetNew: An output that contains what is required to reproduce the plot from generated data
    '''

    # Generate the dataset if none is specified
    if dataset == None:

        # Check if there is a filename. If not, name it "renameMe" and throw a warning
        if filename == None:
            filename = "renameMe"
            print('No fiename specified, has been set to "renameMe".')

        # Set up eta and xi vectors
        etaVec = np.linspace(0, etaHigh, numVec)
        xiVec = np.linspace(0, 1, numVec)

        # Storage matrix
        storeMat = np.zeros((numVec, numVec, M))
        storeNum = np.zeros((numVec, numVec, M))

        # Loop through the parameter space and classify
        for etaInd in range(numVec):
            print('%d out of %d' % (etaInd+1, numVec))
            eta = etaVec[etaInd]
            for xiInd in range(numVec):
                xi = xiVec[xiInd]
                for m in range(M):
                    output = betweenHost(p = p, xi = xi, N = N, n = n, muH = muH, muC = muC, R0 = R0, gammaH = gammaH, gammaC = gammaC, tmax = tmax, eta = eta, epistasis_type = epistasis_type)
                    classOut = classify(output, threshold = threshold)
                    storeMat[xiInd, etaInd, m] = classOut[0]
                    storeNum[xiInd, etaInd, m] = classOut[1]

        # Create a dictionary
        pdict = locals()

        # Save as a pkl
        file = open('%s.pkl' % filename, 'wb')
        pickle.dump(pdict, file)
        file.close()

    # If we just want a plot, load the data
    if dataset != None:

        # Check that the data exists
        file = open('%s.pkl' % dataset, 'rb')
        pdict = pickle.load(file)
        file.close()

    # Take an average of the two storage matrices over the repeats
    storeMatMean = np.mean(pdict['storeMat'], axis=2)
    storeNumMean = np.mean(pdict['storeNum'], axis=2)

    # Extract the three variables required for plotting
    numVec = pdict['numVec']
    etaVec = pdict['etaVec']
    xiVec = pdict['xiVec']

    # Begin a figure
    fig = plt.figure()

    # Add axis properties
    font_size = 18
    plt.rcParams.update({'font.size': font_size})

    # Add axes and fill them
    ax1 = fig.add_subplot(211)
    im1 = ax1.pcolormesh(storeMatMean, cmap='Reds')

    # x-axis
    ax1.set_xticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax1.set_xticklabels(['', '', ''])

    # y-axis
    ax1.set_ylabel(r'Strength of epistasis, $\xi$')
    ax1.set_yticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax1.set_yticklabels([xiVec[0], xiVec[round((numVec-1)/2)], xiVec[-1]])

    # Colorbar
    im1.set_clim(0,20)
    cbar1 = plt.colorbar(im1, extend='max')

    # Add axes and fill them
    ax2 = fig.add_subplot(212)
    im2 = ax2.pcolormesh(storeNumMean, cmap='Greys')

    # x-axis
    ax2.set_xlabel(r'Strength of cross-immunity, $\eta$')
    ax2.set_xticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax2.set_xticklabels([etaVec[0], etaVec[round((numVec-1)/2)], etaVec[-1]])

    # y-axis
    ax2.set_ylabel(r'Strength of epistasis, $\xi$')
    ax2.set_yticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax2.set_yticklabels([xiVec[0], xiVec[round((numVec-1)/2)], xiVec[-1]])

    # Colorbar
    im2.set_clim(0,30)
    cbar2 = plt.colorbar(im2)

    plt.show()

#%% Heat map over proportion immunocompromised and relative infectious period

def heatMapPropInf(dataset = None, filename = None, relInfHigh = 20.0, numVec = 19, M = 10, threshold = 10**(-2), xi = 0.8, N = 1e7, n = 30, muH = 0.01, muC = 0.01, R0 = 3, gammaH = 1/7, tmax = 1460, eta = 5.0, epistasis_type = 'cos'):
    '''
    Code to either plot or generate a heat map over the cross-immunity and epistasis strengths.
    If only a plot is required, specify a dataset which is the output of this function.
    If you need to run dataset, please specify the parameters but leave datasets=None

    INPUTS:
    -------
    dataset: Filename where the output is stored without extension. Should be a .mat file.
    filename: Filename for if a dataset is to be created
    relInfHigh: The maximum value of relative infectious period. It is assumed the lower bound is 1
    numVec: The number in each parameter vector
    M: Number of independent repeats per data point
    threshold: The threshold for the classify function

    xi: double greater than 0 - Strength of epistasis, large values give smaller fitness valleys
    N: int greater than 0 - Number in population
    n: int greater than 0 - Number of phenotypes
    muH: double greater than 0 - Adaption rate for immunocompetent
    muC: double greater than 0 - Adaption rate for immunosupressed
    R0: double greater than 0 - Basic reproductive number
    gammaH: double greater than 0 - Recovery rate of immuno-competent individuals
    tmax: double greater than 0 - Final time (days)
    eta: double greater than 0 - Strength of cross-immunity, large values give low cross immunity
    epistasis_type: string in {'mod', 'quad', 'cos'} - Chooses the type of epistasis valley. 'mod' by default
    
    OUTPUTS:
    --------
    datasetNew: An output that contains what is required to reproduce the plot from generated data
    '''

    # Generate the dataset if none is specified
    if dataset == None:

        # Check if there is a filename. If not, name it "renameMe" and throw a warning
        if filename == None:
            filename = "renameMe"
            print('No fiename specified, has been set to "renameMe".')

        # Set up eta and xi vectors
        pVec = np.linspace(0, 0.1, numVec)
        gammaVec = np.linspace(1, relInfHigh, numVec)

        # Storage matrix
        storeMat = np.zeros((numVec, numVec, M))
        storeNum = np.zeros((numVec, numVec, M))

        # Loop through the parameter space and classify
        for pInd in range(numVec):
            print('%d out of %d' % (pInd+1, numVec))
            p = pVec[pInd]
            for gammaInd in range(numVec):
                gammaRatio = gammaVec[gammaInd]
                for m in range(M):
                    output = betweenHost(p = p, xi = xi, N = N, n = n, muH = muH, muC = muC, R0 = R0, gammaH = gammaH, gammaC = gammaH/gammaRatio, tmax = tmax, eta = eta, epistasis_type = epistasis_type)
                    classOut = classify(output, threshold = threshold)
                    storeMat[gammaInd, pInd, m] = classOut[0]
                    storeNum[gammaInd, pInd, m] = classOut[1]

        # Create a dictionary
        pdict = locals()

        # Save as a pkl
        file = open('%s.pkl' % filename, 'wb')
        pickle.dump(pdict, file)
        file.close()

    # If we just want a plot, load the data
    if dataset != None:

        # Check that the data exists
        file = open('%s.pkl' % dataset, 'rb')
        pdict = pickle.load(file)
        file.close()

    # Take an average of the two storage matrices over the repeats
    storeMatMean = np.mean(pdict['storeMat'], axis=2)
    storeNumMean = np.mean(pdict['storeNum'], axis=2)

    # Extract the three variables required for plotting
    numVec = pdict['numVec']
    pVec = pdict['pVec']
    gammaVec = pdict['gammaVec']

    # Begin a figure
    fig = plt.figure()

    # Add axis properties
    font_size = 18
    plt.rcParams.update({'font.size': font_size})

    # Add axes and fill them
    ax1 = fig.add_subplot(211)
    im1 = ax1.pcolormesh(storeMatMean, cmap='Reds')

    # x-axis
    ax1.set_xticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax1.set_xticklabels(['', '', ''])

    # y-axis
    ax1.set_ylabel(r'Reltive infectious period, $\gamma_H/\gamma_C$')
    ax1.set_yticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax1.set_yticklabels([gammaVec[0], gammaVec[round((numVec-1)/2)], gammaVec[-1]])

    # Colorbar
    im1.set_clim(0,20)
    cbar1 = plt.colorbar(im1, extend='max')

    # Add axes and fill them
    ax2 = fig.add_subplot(212)
    im2 = ax2.pcolormesh(storeNumMean, cmap='Greys')

    # x-axis
    ax2.set_xlabel('Percentage immunocompromised')
    ax2.set_xticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax2.set_xticklabels(['{:.0%}'.format(pVec[0]), '{:.0%}'.format(pVec[round((numVec-1)/2)]), '{:.0%}'.format(pVec[-1])])

    # y-axis
    ax2.set_ylabel(r'Reltive infectious period, $\gamma_H/\gamma_C$')
    ax2.set_yticks([0.5, (numVec-1)/2+0.5, (numVec-1)+0.5])
    ax2.set_yticklabels([gammaVec[0], gammaVec[round((numVec-1)/2)], gammaVec[-1]])

    # Colorbar
    im2.set_clim(0,30)
    cbar2 = plt.colorbar(im2)

    plt.show()

#%% Within-host dynamics

def withinHost(n = 30, r = 1.0, mu = 5e-3, eta = 0.1, kappa = 2e-2, q = 0.1, d = 1e-4, tmax = 150, tau = 0.001):
    '''
    Function which simulates the evolution of a phenotype within an immuno-suppressed individual
    
    INPUTS:
    -------
    n: int greater than 0 - Number of phenotypes
    r: double greater than 0 - Growth rate of variants
    mu: double greater than 0 - Mutation rate
    eta: double greater than 0 - Cross-immunity strength
    kappa: double greater than 0 - Clearance rate by immune response
    q: double greater than 0 - Factor of clearance that the immune repsonse gains
    d: double greater than 0 - Decay rate of immune response
    tmax: double greater than 0 - Final time (days)
    tau: double greater than 0 - Tau-leap step size
    
    OUTUTS:
    -------
    VMat: ndarray - Varint density over time
    RMat: ndarray - Immune response over time
    params: dict - Dictionary of all parameters
    '''

    # Save the parameters
    params = locals()

    # Variant and time vectors
    nVec = np.arange(1, n+1)
    tVec = np.arange(0, tmax, tau)
    nt = len(tVec)

    # Cross immunity matrix X
    X = np.zeros((n,n))
    for ii in range(n):
        for jj in range(n):
            if eta > 0:
                X[ii,jj] = np.exp(-(ii-jj)**2/(2*eta))
            else:
                X[ii,jj] = 1*(ii==jj)

    # Add the cross-immunity to the list of parameters
    params['X'] = X

    # Storage matrices
    VMat = np.zeros((n, nt))
    RMat = np.zeros((n, nt))

    # Initialise
    VVec = np.zeros(n)
    RVec = np.ones(n)
    VVec[0] = 10

    # Store initial values
    VMat[:,0] = VVec
    RMat[:,0] = RVec

    # Loop through time
    for t in range(1, nt):

        # Find the number of each event
        growth = np.random.poisson(r*VVec*tau)
        mutation = np.random.poisson(mu*VVec*tau)
        probs = np.hstack((1, 0.5*np.ones(n-2), 0))  # Probabilities that a mutation is to the right
        mutation_right = np.random.binomial(mutation, probs)
        mutation_left = mutation - mutation_right
        clearance = np.random.poisson(kappa*np.matmul(X, RVec)*VVec*tau)
        creation = np.random.poisson(kappa*q*VVec*RVec*tau)
        decay = np.random.poisson(d*RVec*tau)

        # Update numbers
        VVec = VVec + growth - mutation + np.hstack((0, mutation_right[0:-1])) + np.hstack((mutation_left[1:], 0)) - clearance
        RVec = RVec + creation - decay

        # Store in the matrix
        VMat[:,t] = VVec
        RMat[:,t] = RVec

    return([VMat, RMat, params])
