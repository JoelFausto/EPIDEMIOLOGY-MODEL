"""
Programa principal para simulação de modelos epidemiológicos.
Simula SIR, SEIR e modelo SEIC para Doença de Chagas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from modelos_epidemiologicos import (
    sir_model, seir_model, host_vector_chagas_improved,
    check_population_conservation, calculate_basic_reproduction_number,
    get_default_chagas_params
)

def run_sir_simulation(N_human, t):
    """Executa simulação do modelo SIR."""
    print("Simulando Modelo SIR...")
    
    # Parâmetros
    beta_sir = 0.3          
    gamma_sir = 1/60        # Recuperação média 60 dias
    
    # Condições iniciais normalizadas
    I0_sir = 10/N_human
    R0_sir = 0
    S0_sir = 1.0 - I0_sir - R0_sir

    y0_sir = [S0_sir, I0_sir, R0_sir]
    sol_sir = odeint(sir_model, y0_sir, t, args=(beta_sir, gamma_sir))
    S_sir, I_sir, R_sir = sol_sir.T
    
    # Converter para números absolutos
    S_sir_abs = S_sir * N_human
    I_sir_abs = I_sir * N_human
    R_sir_abs = R_sir * N_human
    
    check_population_conservation(sol_sir, "SIR")
    R0_sir_val = calculate_basic_reproduction_number((beta_sir, gamma_sir), "SIR")
    print(f"R0 estimado: {R0_sir_val:.2f}")
    
    return S_sir_abs, I_sir_abs, R_sir_abs, R0_sir_val

def run_seir_simulation(N_human, t):
    """Executa simulação do modelo SEIR."""
    print(" Simulando Modelo SEIR...")
    
    # Parâmetros
    beta_seir = 0.25
    sigma_seir = 1/14       # Incubação 14 dias
    gamma_seir = 1/60

    # Condições iniciais normalizadas
    I0_seir = 10/N_human
    E0_seir = 0
    R0_seir = 0
    S0_seir = 1.0 - I0_seir - E0_seir - R0_seir

    y0_seir = [S0_seir, E0_seir, I0_seir, R0_seir]
    sol_seir = odeint(seir_model, y0_seir, t, args=(beta_seir, sigma_seir, gamma_seir))
    S_seir, E_seir, I_seir, R_seir = sol_seir.T
    
    # Converter para números absolutos
    S_seir_abs = S_seir * N_human
    E_seir_abs = E_seir * N_human
    I_seir_abs = I_seir * N_human
    R_seir_abs = R_seir * N_human
    
    check_population_conservation(sol_seir, "SEIR")
    R0_seir_val = calculate_basic_reproduction_number((beta_seir, sigma_seir, gamma_seir), "SEIR")
    print(f"R0 estimado: {R0_seir_val:.2f}")
    
    return S_seir_abs, E_seir_abs, I_seir_abs, R_seir_abs, R0_seir_val

def run_chagas_simulation(N_human, N_animals, t):
    """Executa simulação do modelo de Chagas."""
    print("Simulando Modelo SEIC Host-Vector-Animal...")
    
    # Parâmetros e condições iniciais
    params_chagas = get_default_chagas_params(N_human, N_animals)
    
    I_a0 = 50; S_a0 = N_animals - I_a0 
    S_v0 = 10000; E_v0 = 0; I_v0 = 10
    I_h0 = 10; E_h0 = 0; C_h0 = 0; S_h0 = N_human - I_h0 - E_h0 - C_h0
    
    y0_chagas = [S_h0, E_h0, I_h0, C_h0, S_v0, E_v0, I_v0, S_a0, I_a0]
    sol_chagas = odeint(host_vector_chagas_improved, y0_chagas, t, args=(params_chagas,))
    S_h, E_h, I_h, C_h, S_v, E_v, I_v, S_a, I_a = sol_chagas.T
    
    R0_chagas_val = calculate_basic_reproduction_number(params_chagas, "CHAGAS")
    print(f"   R0 estimado: {R0_chagas_val:.2f}")
    
    return S_h, E_h, I_h, C_h, S_v, E_v, I_v, S_a, I_a, R0_chagas_val

def plot_results(t, sir_data, seir_data, chagas_data):
    """Plota os resultados das simulações."""
    print("\nGerando gráficos...")
    
    S_sir_abs, I_sir_abs, R_sir_abs, R0_sir = sir_data
    S_seir_abs, E_seir_abs, I_seir_abs, R_seir_abs, R0_seir = seir_data
    S_h, E_h, I_h, C_h, S_v, E_v, I_v, S_a, I_a, R0_chagas = chagas_data
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig.suptitle("Comparação de Modelos Epidemiológicos para Doença de Chagas", 
                 fontsize=16, fontweight='bold')

    # GRÁFICO 1: Modelo SIR
    axes[0].plot(t, S_sir_abs, label="Suscetíveis (S)", color='blue', linewidth=2, alpha=0.8)
    axes[0].plot(t, I_sir_abs, label="Infectados (I)", color='red', linewidth=3)
    axes[0].plot(t, R_sir_abs, label="Recuperados (R)", color='green', linewidth=2, alpha=0.8)
    axes[0].set_title(f"1. Modelo SIR Clássico (R₀ = {R0_sir:.2f})", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("População", fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].set_ylim(bottom=0)

    # GRÁFICO 2: Modelo SEIR
    axes[1].plot(t, S_seir_abs, label="Suscetíveis (S)", color='blue', linewidth=2, alpha=0.8)
    axes[1].plot(t, E_seir_abs, label="Expostos (E)", color='orange', linewidth=2, alpha=0.8)
    axes[1].plot(t, I_seir_abs, label="Infectados (I)", color='red', linewidth=3)
    axes[1].plot(t, R_seir_abs, label="Recuperados (R)", color='green', linewidth=2, alpha=0.8)
    axes[1].set_title(f"2. Modelo SEIR com Fase de Incubação (R₀ = {R0_seir:.2f})", 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel("População", fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].set_ylim(bottom=0)

    # GRÁFICO 3: Modelo SEIC Host-Vector-Animal
    axes[2].plot(t, I_h, label="Humanos Agudos (I_h)", color='red', linewidth=3)
    axes[2].plot(t, C_h, label="Humanos Crônicos (C_h)", color='purple', linewidth=3)
    axes[2].plot(t, I_v, label="Vetores Infectados (I_v)", color='darkgreen', linestyle='--', linewidth=2)
    axes[2].plot(t, I_a, label="Animais Infectados (I_a)", color='darkorange', linestyle=':', linewidth=2)
    axes[2].set_title(f"3. Modelo SEIC Host-Vector-Animal para Doença de Chagas (R₀ = {R0_chagas:.2f})", 
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Dias", fontsize=12)
    axes[2].set_ylabel("População", fontsize=12)
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    return S_h, E_h, I_h, C_h, I_sir_abs, I_seir_abs

def plot_comparison(t, I_sir_abs, I_seir_abs, I_h):
    """Plota gráfico comparativo dos infectados."""
    plt.figure(figsize=(12, 6))
    plt.plot(t, I_sir_abs, label='SIR', color='red', linewidth=2, alpha=0.7)
    plt.plot(t, I_seir_abs, label='SEIR', color='darkred', linewidth=2, alpha=0.7)
    plt.plot(t, I_h, label='Chagas (Agudos)', color='crimson', linewidth=3)
    plt.title('Comparação: Curva de Infectados entre Modelos', fontsize=14, fontweight='bold')
    plt.xlabel('Dias')
    plt.ylabel('População Infectada')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

def print_summary(I_sir_abs, I_seir_abs, I_h, C_h, I_v, I_a, t):
    """Imprime resumo das simulações."""
    print("\n" + "="*60)
    print("RESUMO DA SIMULAÇÃO")
    print("="*60)
    
    # Picos de infecção
    peak_sir = np.max(I_sir_abs)
    peak_seir = np.max(I_seir_abs)
    peak_chagas_h = np.max(I_h)
    
    print(f"Pico de Infectados:")
    print(f"- SIR:  {peak_sir:,.0f} pessoas (dia {np.argmax(I_sir_abs)})")
    print(f"- SEIR: {peak_seir:,.0f} pessoas (dia {np.argmax(I_seir_abs)})")
    print(f"- Chagas (agudos): {peak_chagas_h:,.0f} pessoas (dia {np.argmax(I_h)})")
    
    print(f"\nCrônicos ao final (Chagas): {C_h[-1]:,.0f} pessoas")
    print(f"Vetores infectados ao final: {I_v[-1]:.0f}")
    print(f"Animais infectados ao final: {I_a[-1]:.0f}")

def main():
    """Função principal do programa."""
    print("Iniciando simulações...")
    
    # --- Configurações de Simulação ---
    N_human = 1_000_000          # População humana total
    N_animals = 50_000           # População de animais reservatórios
    tempo_simulacao = 365 * 3    # 3 anos em dias
    t = np.linspace(0, tempo_simulacao, tempo_simulacao + 1)
    
    # --- Executar Simulações ---
    sir_results = run_sir_simulation(N_human, t)
    seir_results = run_seir_simulation(N_human, t)
    chagas_results = run_chagas_simulation(N_human, N_animals, t)
    
    # --- Plotar Resultados ---
    final_data = plot_results(t, sir_results, seir_results, chagas_results)
    S_h, E_h, I_h, C_h, I_sir_abs, I_seir_abs = final_data
    
    # --- Gráfico Comparativo ---
    plot_comparison(t, I_sir_abs, I_seir_abs, I_h)
    
    # --- Resumo ---
    print_summary(I_sir_abs, I_seir_abs, I_h, C_h, chagas_results[7], chagas_results[8], t)
    
    print("Simulações concluídas com sucesso!")

if __name__ == "__main__":
    main()