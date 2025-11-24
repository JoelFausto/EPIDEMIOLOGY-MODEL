"""
Módulo com modelos epidemiológicos para simulação de doenças.
Inclui SIR, SEIR e modelo SEIC para Doença de Chagas.
"""

import numpy as np
from scipy.integrate import odeint

# =======================================================
# 1. MODELO SIR (Susceptível, Infectado, Recuperado/Removido)
# =======================================================
def sir_model(y, t, beta, gamma):
    """
    Modelo SIR Clássico com população normalizada.
    
    Parâmetros:
    -----------
    y : list
        [S, I, R] - Estados normalizados (valores entre 0 e 1)
    t : array
        Array de tempo
    beta : float
        Taxa de transmissão
    gamma : float
        Taxa de recuperação
    
    Retorna:
    --------
    list : [dSdt, dIdt, dRdt]
    """
    S, I, R = y
    N = S + I + R  # População total normalizada
    
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]

# =======================================================
# 2. MODELO SEIR (Susceptível, Exposto, Infectado, Recuperado)
# =======================================================
def seir_model(y, t, beta, sigma, gamma):
    """
    Modelo SEIR Clássico com população normalizada.
    
    Parâmetros:
    -----------
    y : list
        [S, E, I, R] - Estados normalizados (valores entre 0 e 1)
    t : array
        Array de tempo
    beta : float
        Taxa de transmissão
    sigma : float
        Taxa de progressão de exposto para infectado
    gamma : float
        Taxa de recuperação
    
    Retorna:
    --------
    list : [dSdt, dEdt, dIdt, dRdt]
    """
    S, E, I, R = y
    N = S + E + I + R  # População total normalizada
    
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return [dSdt, dEdt, dIdt, dRdt]

# =======================================================
# 3. MODELO SEIC HOST–VECTOR COM RESERVATÓRIO ANIMAL
# (Melhorado para Doença de Chagas)
# =======================================================
def host_vector_chagas_improved(y, t, params):
    """
    Modelo acoplado Humano (SEIC) + Vetor (SEI) + Animal (SI).
    
    Parâmetros:
    -----------
    y : list
        [S_h, E_h, I_h, C_h, S_v, E_v, I_v, S_a, I_a] - Estados em números absolutos
    t : array
        Array de tempo
    params : dict
        Dicionário com todos os parâmetros do modelo
    
    Retorna:
    --------
    list : [dS_h, dE_h, dI_h, dC_h, dS_v, dE_v, dI_v, dS_a, dI_a]
    """
    S_h, E_h, I_h, C_h, S_v, E_v, I_v, S_a, I_a = y

    # Parâmetros
    beta_hv = params['beta_hv']  # Vetor -> Humano
    beta_vh = params['beta_vh']  # Humano -> Vetor
    beta_va = params['beta_va']  # Animal -> Vetor
    beta_av = params['beta_av']  # Vetor -> Animal
    sigma_h = params['sigma_h']  # incubação humana (E -> I)
    sigma_v = params['sigma_v']  # incubação vetorial (E -> I)
    gamma_h = params['gamma_h']  # cronificação humana (I -> C)
    mu_v = params['mu_v']        # mortalidade do vetor
    Lambda_v = params['Lambda_v']# nascimento do vetor
    mu_a = params['mu_a']        # mortalidade do animal
    Lambda_a = params['Lambda_a']# nascimento do animal
    N_h = params['N_h']
    N_a = params['N_a']

    # Força de Infecção (Lambda)
    total_vectors = S_v + E_v + I_v
    lambda_h = beta_hv * I_v / total_vectors if total_vectors > 0 else 0
    lambda_v = (beta_vh * I_h / N_h) + (beta_va * I_a / N_a)
    lambda_a = beta_av * I_v / total_vectors if total_vectors > 0 else 0

    # Dinâmica Humana (SEIC) - População fechada
    dS_h = -lambda_h * S_h
    dE_h = lambda_h * S_h - sigma_h * E_h
    dI_h = sigma_h * E_h - gamma_h * I_h
    dC_h = gamma_h * I_h 

    # Dinâmica Vetorial (SEI) - População aberta
    dS_v = Lambda_v - lambda_v * S_v - mu_v * S_v
    dE_v = lambda_v * S_v - sigma_v * E_v - mu_v * E_v
    dI_v = sigma_v * E_v - mu_v * I_v

    # Dinâmica Animal (SI) - População aberta
    dS_a = Lambda_a - lambda_a * S_a - mu_a * S_a
    dI_a = lambda_a * S_a - mu_a * I_a

    return [dS_h, dE_h, dI_h, dC_h, dS_v, dE_v, dI_v, dS_a, dI_a]

# =======================================================
# FUNÇÕES AUXILIARES
# =======================================================
def check_population_conservation(sol, model_name, tolerance=1e-6):
    """
    Verifica se a população total é conservada nos modelos fechados.
    
    Parâmetros:
    -----------
    sol : array
        Solução da EDO
    model_name : str
        Nome do modelo para mensagem
    tolerance : float
        Tolerância para variação populacional
    """
    total_pop = np.sum(sol, axis=1)
    variation = np.max(np.abs(total_pop - total_pop[0]))
    
    if variation > tolerance:
        print(f"Aviso: Variação de população em {model_name}: {variation:.2e}")
    else:
        print(f"População conservada em {model_name}: variação {variation:.2e}")

def calculate_basic_reproduction_number(params, model_type):
    """
    Calcula R0 para diferentes modelos epidemiológicos.
    
    Parâmetros:
    -----------
    params : tuple ou dict
        Parâmetros do modelo
    model_type : str
        Tipo do modelo ('SIR', 'SEIR', 'CHAGAS')
    
    Retorna:
    --------
    float : Número reprodutivo básico R0
    """
    if model_type == "SIR":
        beta, gamma = params
        return beta / gamma
    elif model_type == "SEIR":
        beta, sigma, gamma = params
        return beta / gamma
    elif model_type == "CHAGAS":
        # R0 aproximado para modelo complexo de Chagas
        beta_hv, beta_vh, sigma_v, mu_v, gamma_h = (
            params['beta_hv'], params['beta_vh'], params['sigma_v'], 
            params['mu_v'], params['gamma_h']
        )
        return (beta_hv * beta_vh * sigma_v) / (mu_v * (sigma_v + mu_v) * gamma_h)
    return None

def get_default_chagas_params(N_human, N_animals):
    """
    Retorna parâmetros padrão para o modelo de Chagas.
    
    Parâmetros:
    -----------
    N_human : int
        População humana total
    N_animals : int
        População animal total
    
    Retorna:
    --------
    dict : Parâmetros do modelo de Chagas
    """
    return {
        'beta_hv': 0.00003,   # Vetor -> Humano
        'beta_vh': 0.00005,   # Humano -> Vetor  
        'beta_va': 0.00008,   # Animal -> Vetor
        'beta_av': 0.000005,  # Vetor -> Animal
        'sigma_h': 1/14,      # Incubação humana (14 dias)
        'sigma_v': 1/10,      # Incubação vetorial (10 dias)
        'gamma_h': 1/60,      # Cronificação humana (60 dias)
        'mu_v': 1/60,         # Mortalidade do vetor (60 dias de vida)
        'Lambda_v': 50,       # Nascimento de vetores por dia
        'mu_a': 1/(365*3),    # Mortalidade animal (vida média 3 anos)
        'Lambda_a': N_animals/(365*3), # Natalidade animal
        'N_h': N_human,
        'N_a': N_animals
    }