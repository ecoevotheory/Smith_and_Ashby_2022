README file - Antigenic evolution of SARS-CoV-2 in immunocompromised hosts

--------------------------------------------------------------------------------------------

Contact information:
Name: Cameron Smith
Email: cs640@bath.ac.uk

--------------------------------------------------------------------------------------------

This is the README file for the various parts of code that can used to reproduce data. Note that only the heatmaps have plotting tools built into them.

All code was written and executed in Python 3.8.8.

--------------------------------------------------------------------------------------------

FUNCTION: betweenHost(p = 0.05, xi = 0.8, N = 1e7, n = 30, muH = 0.01, muC = 0.01, R0 = 3, gammaH = 1/7, gammaC = 1/140, tmax = 1460, eta = 5.0, epistasis_type = 'cos')

Simulates a single repeat of the between-host model. All parameters above are the default parameters for a simulation. Code outputs the susceptible and infected populations over time and variant index.

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

--------------------------------------------------------------------------------------------

FUNCTION: classify(betweenHost_output, threshold = 10**(-2)):

Classifies the output of a single betweenHost simulation. As an input, it requires this output list.

INPUTS:
-------
    betweenHost_output: Output from a betweenHost function call to be classified
    threhold: Threshold value below which counts are considered 0

OUTPUTS:
--------
    max(L): The maximum distance between observed variants
    numVar: The total number of observed variants
    '''

--------------------------------------------------------------------------------------------

FUNCTION: heatMapCrossEpi(dataset = None, filename = None, etaHigh = 10.0, numVec = 21, M = 10, threshold = 10**(-2), p = 0.05, N = 1e7, n = 30, muH = 0.01, muC = 0.01, R0 = 3, gammaH = 1/7, gammaC = 1/140, tmax = 1460, epistasis_type = 'cos')

Plotting and data generation code for the cross-immunity and epistasis heat map. This code serves two purposes:
	> If dataset is not specified, then the code will generate a heat map with the specified parameters and will save the output as filename.pkl.
	> If dataset is specified, the code will ignore all other inputs and will instead generate the heat map using the data stored in dataset.pkl.

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

--------------------------------------------------------------------------------------------

FUNCTION: heatMapPropInf(dataset = None, filename = None, relInfHigh = 20.0, numVec = 19, M = 10, threshold = 10**(-2), xi = 0.8, N = 1e7, n = 30, muH = 0.01, muC = 0.01, R0 = 3, gammaH = 1/7, tmax = 1460, eta = 5.0, epistasis_type = 'cos')

Plotting and data generation code for the proportion immunocompromised and relative infectious period heat map. This code serves two purposes:
	> If dataset is not specified, then the code will generate a heat map with the specified parameters and will save the output as filename.pkl.
	> If dataset is specified, the code will ignore all other inputs and will instead generate the heat map using the data stored in dataset.pkl.

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

--------------------------------------------------------------------------------------------

FUNCTION: withinHost(n = 30, r = 1.0, mu = 5e-3, eta = 0.1, kappa = 2e-2, q = 0.1, d = 1e-4, tmax = 150, tau = 0.001)

Simulates a single repeat of the within-host model. All parameters above are the default parameters for a simulation. Code outputs the variant population and immune response over time and variant index.

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