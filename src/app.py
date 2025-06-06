from flask import Flask, render_template, Response, send_file
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import numpy as np
import time
import matplotlib.colors as mcolors
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import tempfile


app = Flask(__name__)

def calcular_campos_e_potencial(X, Y, config_sim):
    """
    Calcula o campo elétrico total (Ex, Ey), o potencial elétrico (V),
    a magnitude do campo elétrico e as máscaras dos materiais.
    """
    E_externo = config_sim['E_externo']
    material_x_min = config_sim['material_x_min']
    material_x_max = config_sim['material_x_max']
    material_y_min = config_sim['material_y_min']
    material_y_max = config_sim['material_y_max']
    tipo_material = config_sim['tipo_material']
    # Removido constante_dieletrica_relativa (usaremos a do isolante)

    # Parâmetros do isolante
    isolante_espessura = config_sim['isolante_espessura']
    constante_dieletrica_isolante = config_sim['constante_dieletrica_isolante']

    # Inicializa campos com o campo externo
    Ex_total = np.full(X.shape, E_externo[0])
    Ey_total = np.full(Y.shape, E_externo[1])

    # Potencial inicial devido ao campo externo
    V_potencial = -(E_externo[0] * X + E_externo[1] * Y)

    # Define as fronteiras do condutor (baseado no 'material')
    condutor_x_min = material_x_min
    condutor_x_max = material_x_max
    condutor_y_min = material_y_min
    condutor_y_max = material_y_max

    # Define as fronteiras do isolante (revestimento)
    isolante_x_min = condutor_x_min - isolante_espessura
    isolante_x_max = condutor_x_max + isolante_espessura
    isolante_y_min = condutor_y_min - isolante_espessura
    isolante_y_max = condutor_y_max + isolante_espessura

    # Máscara booleana para identificar pontos dentro do condutor
    mascara_condutor = (X >= condutor_x_min) & (X <= condutor_x_max) & \
                       (Y >= condutor_y_min) & (Y <= condutor_y_max)

    # Máscara booleana para identificar pontos dentro do isolante (incluindo o condutor)
    mascara_isolante_total = (X >= isolante_x_min) & (X <= isolante_x_max) & \
                             (Y >= isolante_y_min) & (Y <= isolante_y_max)

    # Máscara booleana para identificar pontos *apenas* no isolante
    mascara_isolante_apenas = mascara_isolante_total & ~mascara_condutor

    # Se o material for condutor (agora sempre será, com revestimento)
    if tipo_material == 'condutor':
        # Campo dentro do isolante
        Ex_interno_iso = E_externo[0] / constante_dieletrica_isolante
        Ey_interno_iso = E_externo[1] / constante_dieletrica_isolante
        Ex_total[mascara_isolante_apenas] = Ex_interno_iso
        Ey_total[mascara_isolante_apenas] = Ey_interno_iso

        # Campo dentro do condutor
        Ex_total[mascara_condutor] = 0.0
        Ey_total[mascara_condutor] = 0.0

        # Potencial dentro do condutor (aproximado pelo centro e campo externo)
        x_centro_cond = (condutor_x_min + condutor_x_max) / 2
        y_centro_cond = (condutor_y_min + condutor_y_max) / 2
        V_const_condutor = -(E_externo[0] * x_centro_cond + E_externo[1] * y_centro_cond)
        V_potencial[mascara_condutor] = V_const_condutor

        # --- Desafio: Potencial no Isolante ---
        # A_original_code não lida bem com isso. Uma_simplificação_é_manter_o_potencial_externo
        # e_confiar_no_contour_para_lidar_com_a_descontinuidade_no_condutor.
        # Uma_melhoria_seria_integrar_o_campo_no_isolante,_mas_é_complexo.
        # Por_ora,_não_modificaremos_V_no_isolante,_aceitando_a_imprecisão_visual.

    # Se fosse dielétrico (mantido para referência, mas não usado na nova lógica)
    # elif tipo_material == 'dielétrico':
    #     ... (código original para dielétrico)

    magnitude_E = np.sqrt(Ex_total**2 + Ey_total**2)
    # Retorna as máscaras para visualização
    return Ex_total, Ey_total, V_potencial, magnitude_E, mascara_condutor, mascara_isolante_total

def visualizar_frame(ax, X, Y, Ex_total, Ey_total, V_potencial, magnitude_E, mascara_condutor, mascara_isolante_total, config_sim):
    """
    Visualiza UM frame da simulação (adaptada para ser chamada na animação).
    Recebe 'ax' (axes) para desenhar.
    """
    ax.clear() # Limpa o frame anterior

    x_min, x_max = config_sim['x_min'], config_sim['x_max']
    y_min, y_max = config_sim['y_min'], config_sim['y_max']
    tipo_material = config_sim['tipo_material']
    resolucao = config_sim['resolucao']
    E_externo = config_sim['E_externo']
    constante_dieletrica_isolante = config_sim['constante_dieletrica_isolante']
    isolante_espessura = config_sim['isolante_espessura']

    # --- Obter as fronteiras atuais (já calculadas em 'update') ---
    material_x_min = config_sim['material_x_min']
    material_x_max = config_sim['material_x_max']
    material_y_min = config_sim['material_y_min']
    material_y_max = config_sim['material_y_max']

    condutor_x_min = material_x_min
    condutor_x_max = material_x_max
    condutor_y_min = material_y_min
    condutor_y_max = material_y_max

    isolante_x_min = condutor_x_min - isolante_espessura
    isolante_x_max = condutor_x_max + isolante_espessura
    isolante_y_min = condutor_y_min - isolante_espessura
    isolante_y_max = condutor_y_max + isolante_espessura

    # --- Desenha o isolante ---
    isolante_rect = Rectangle((isolante_x_min, isolante_y_min),
                              isolante_x_max - isolante_x_min,
                              isolante_y_max - isolante_y_min,
                              linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.5,
                              label=rf'Isolante ($\epsilon_r = {constante_dieletrica_isolante}$)')
    ax.add_patch(isolante_rect)

    # --- Desenha o condutor ---
    condutor_rect = Rectangle((condutor_x_min, condutor_y_min),
                              condutor_x_max - condutor_x_min,
                              condutor_y_max - condutor_y_min,
                              linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.9,
                              label='Condutor')
    ax.add_patch(condutor_rect)


    # --- Linhas Equipotenciais ---
    num_equipotenciais = 20
    # Ajusta os limites para considerar ambas as componentes do campo, se existirem
    v_lim_x_min, v_lim_x_max = - (E_externo[0] * config_sim['x_max']), - (E_externo[0] * config_sim['x_min'])
    v_lim_y_min, v_lim_y_max = - (E_externo[1] * config_sim['y_max']), - (E_externo[1] * config_sim['y_min'])
    v_lim_min = min(v_lim_x_min, v_lim_y_min)
    v_lim_max = max(v_lim_x_max, v_lim_y_max)
    if v_lim_min > v_lim_max: v_lim_min, v_lim_max = v_lim_max, v_lim_min
    if np.isclose(v_lim_min, v_lim_max): # Garante que haja um intervalo
        v_lim_min -= 1
        v_lim_max += 1
    v_lim_min -= abs(v_lim_max-v_lim_min)*0.2
    v_lim_max += abs(v_lim_max-v_lim_min)*0.2
    niveis_V = np.linspace(v_lim_min, v_lim_max, num_equipotenciais)
    # Desenha o contorno *excluindo* o interior do condutor para evitar artefatos
    V_plot = V_potencial.copy()
    # V_plot[mascara_condutor] = np.nan # Opção: Não desenhar dentro do condutor
    contour = ax.contour(X, Y, V_plot, levels=niveis_V, colors='purple', linewidths=1.0, linestyles='--', alpha=0.8)
    ax.clabel(contour, inline=True, fontsize=9, fmt='%1.1f V')

    # --- Linhas de Campo Elétrico (Streamplot) ---
    max_e_abs = np.max(np.abs(E_externo))
    min_mag, max_mag = 0, max_e_abs * 2.0 # Aumenta um pouco o max para melhor visualização da cor
    if max_mag < 1e-6: max_mag = 1.0 # Evita max_mag zero
    use_color_for_magnitude = not np.isclose(min_mag, max_mag)
    stream_color = magnitude_E if use_color_for_magnitude else 'black'
    cmap_streamplot = 'viridis'
    ax.streamplot(X, Y, Ex_total, Ey_total, density=1.8, color=stream_color,
                  linewidth=1.0, cmap=cmap_streamplot, arrowsize=1.2, arrowstyle='->',
                  norm=mcolors.Normalize(vmin=min_mag, vmax=max_mag))

    # --- Efeitos Específicos do Material (Cargas Induzidas no CONDUTOR) ---
    if tipo_material == 'condutor':
        carga_offset_visual = resolucao * 3.0 # Aumenta o offset para o texto
        tamanho_carga_visual = 15 # Aumenta o tamanho para o sinal
        tamanho_valor_visual = 10 # Tamanho para o valor

        x_centro_cond = (condutor_x_min + condutor_x_max) / 2
        y_centro_cond = (condutor_y_min + condutor_y_max) / 2

        # Cargas devido a Ex
        if abs(E_externo[0]) > 1e-6:
            sinal_dir, sinal_esq = ('+', '-') if E_externo[0] > 0 else ('-', '+')
            valor_carga = abs(E_externo[0])
            # Esquerda
            ax.text(condutor_x_min - carga_offset_visual, y_centro_cond, f"{sinal_esq}",
                    color='red', fontsize=tamanho_carga_visual, ha='center', va='bottom', fontweight='bold', zorder=10)
            ax.text(condutor_x_min - carga_offset_visual, y_centro_cond, rf"$\sigma \propto {valor_carga:.1f}$",
                    color='black', fontsize=tamanho_valor_visual, ha='center', va='top', zorder=10)
            # Direita
            ax.text(condutor_x_max + carga_offset_visual, y_centro_cond, f"{sinal_dir}",
                    color='red', fontsize=tamanho_carga_visual, ha='center', va='bottom', fontweight='bold', zorder=10)
            ax.text(condutor_x_max + carga_offset_visual, y_centro_cond, rf"$\sigma \propto {valor_carga:.1f}$",
                    color='black', fontsize=tamanho_valor_visual, ha='center', va='top', zorder=10)

        # Cargas devido a Ey
        if abs(E_externo[1]) > 1e-6:
            sinal_sup, sinal_inf = ('+', '-') if E_externo[1] > 0 else ('-', '+')
            valor_carga = abs(E_externo[1])
            # Inferior
            ax.text(x_centro_cond, condutor_y_min - carga_offset_visual, f"{sinal_inf}",
                    color='red', fontsize=tamanho_carga_visual, ha='center', va='center', fontweight='bold', zorder=10)
            ax.text(x_centro_cond, condutor_y_min - carga_offset_visual * 1.8 , rf"$\sigma \propto {valor_carga:.1f}$",
                    color='black', fontsize=tamanho_valor_visual, ha='center', va='center', zorder=10)
            # Superior
            ax.text(x_centro_cond, condutor_y_max + carga_offset_visual, f"{sinal_sup}",
                    color='red', fontsize=tamanho_carga_visual, ha='center', va='center', fontweight='bold', zorder=10)
            ax.text(x_centro_cond, condutor_y_max + carga_offset_visual * 1.8 , rf"$\sigma \propto {valor_carga:.1f}$",
                    color='black', fontsize=tamanho_valor_visual, ha='center', va='center', zorder=10)


    # --- Configurações finais do plot ---
    ax.set_title(f'Condutor Revestido em Campo Elétrico Variável', fontsize=14)
    ax.set_xlabel('Posição x', fontsize=12)
    ax.set_ylabel('Posição y', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Legenda
    legend_elements = [Rectangle((0, 0), 1, 1, color='lightblue', label=rf'Isolante ($\epsilon_r = {constante_dieletrica_isolante}$)'),
                       Rectangle((0, 0), 1, 1, color='lightgray', label='Condutor'),
                       Line2D([0], [0], color='purple', lw=1, linestyle='--', label='Equipotenciais'),
                       Line2D([0], [0], color='gray', lw=1.5, label='Linhas de Campo'),
                       Line2D([0], [0], marker='$+$', color='red', label='Cargas Induzidas ($\sigma$)', linestyle='None')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)


# --- Configurações Principais da Simulação ---
def obter_configuracoes_simulacao():
    config = {
        'x_min': -7, 'x_max': 7,
        'y_min': -5, 'y_max': 5,
        'resolucao': 0.2,

        'material_x_min': -2, 'material_x_max': 2, # Posição inicial do CONDUTOR
        'material_y_min': -1, 'material_y_max': 1, # Posição inicial do CONDUTOR

        'tipo_material': 'condutor', # Agora sempre será condutor revestido
        'isolante_espessura': 0.5, # Espessura da camada isolante
        'constante_dieletrica_isolante': 4.0, # Constante dielétrica do isolante

        'E_externo': np.array([1.0, 1.0]) # Campo inicial
    }
    return config

# --- Execução da Animação ---
if __name__ == '__main__':
    config_sim = obter_configuracoes_simulacao()

    # Geração da Grade
    x_pts = np.arange(config_sim['x_min'], config_sim['x_max'] + config_sim['resolucao'], config_sim['resolucao'])
    y_pts = np.arange(config_sim['y_min'], config_sim['y_max'] + config_sim['resolucao'], config_sim['resolucao'])
    X, Y = np.meshgrid(x_pts, y_pts)

    # Configuração da Figura e Animação
    fig, ax = plt.subplots(figsize=(13, 10))

    # --- Configuração da Colorbar (fora da animação) ---
    max_e_abs = np.max(np.abs(config_sim['E_externo'])) * 2.0 # Ajusta escala
    min_mag, max_mag = 0, max_e_abs * 2.0 if max_e_abs > 1e-6 else 1.0
    cmap_streamplot = 'viridis'
    norm = mcolors.Normalize(vmin=min_mag, vmax=max_mag)
    sm = plt.cm.ScalarMappable(cmap=cmap_streamplot, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='Magnitude do Campo Elétrico (E)', pad=0.02, aspect=30, shrink=0.8)


    num_frames = 100
    largura_condutor = config_sim['material_x_max'] - config_sim['material_x_min']
    x_inicio = -5.0
    x_fim = 5.0
    posicoes_x_centro = np.concatenate([
        np.linspace(x_inicio, x_fim, num_frames // 2),
        np.linspace(x_fim, x_inicio, num_frames // 2)
    ])

    Ex0 = 2.0 # Campo X inicial
    Exf = -2.0 # Campo X final
    Ey0 = 0.0 # Campo Y inicial
    Eyf = 2.0 # Campo Y final

    Ex = np.linspace(Ex0, Exf, num_frames)
    Ey = np.linspace(Ey0, Eyf, num_frames)


    def update(frame):
        """Função que atualiza cada frame da animação."""
        print(f"Gerando frame {frame+1}/{num_frames}")

        # Atualiza a posição do centro do condutor
        x_centro_atual = posicoes_x_centro[frame]
        config_sim['material_x_min'] = x_centro_atual - largura_condutor / 2
        config_sim['material_x_max'] = x_centro_atual + largura_condutor / 2
        # Atualiza o campo externo
        config_sim['E_externo'] = np.array([Ex[frame], Ey[frame]])

        # Recalcula tudo para o frame atual
        Ex_total, Ey_total, V_potencial, magnitude_E, mascara_condutor, mascara_isolante_total = \
            calcular_campos_e_potencial(X, Y, config_sim)

        # Atualiza a colorbar se necessário (o max pode mudar com E)
        max_e_abs = np.max(np.abs(config_sim['E_externo']))
        max_mag_atual = max_e_abs * 2.0 if max_e_abs > 1e-6 else 1.0
        cbar.mappable.set_norm(mcolors.Normalize(vmin=0, vmax=max_mag_atual))


        # Desenha o frame
        visualizar_frame(ax, X, Y, Ex_total, Ey_total, V_potencial, magnitude_E,
                         mascara_condutor, mascara_isolante_total, config_sim)
        return ax,


    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=100, # ms por frame
                                  blit=False,   # Blit=False é mais seguro com contour/colorbar
                                  repeat=True)

    plt.tight_layout(rect=[0, 0, 0.90, 1]) # Ajusta layout para a colorbar
    plt.show()

    # Para salvar (opcional, requer ffmpeg ou similar):
    # print("Salvando animação...")
    # ani.save('condutor_revestido_animacao.mp4', writer='ffmpeg', fps=10, dpi=150)
    # print("Animação salva.")
    
    
# --- Configurações da Simulação ---
config_sim = {
    'x_min': -5, 'x_max': 5,
    'y_min': -5, 'y_max': 5,
    'resolucao': 0.2,
    'material_x_min': -1,
    'material_x_max': 1,
    'material_y_min': -2,
    'material_y_max': 2,
    'tipo_material': 'condutor',
    'isolante_espessura': 0.5,
    'constante_dieletrica_isolante': 3.0,
    'E_externo': [0.0, 0.0],  # Será atualizado dinamicamente
}

# Geração da malha
x = np.arange(config_sim['x_min'], config_sim['x_max'], config_sim['resolucao'])
y = np.arange(config_sim['y_min'], config_sim['y_max'], config_sim['resolucao'])
X, Y = np.meshgrid(x, y)

# --- Valores de campo externo para cada frame ---
valores_Ex = np.linspace(-3, 3, 30)  # Pode ajustar quantos frames quiser
valores_Ey = np.zeros_like(valores_Ex)  # Campo variando apenas em Ex (horizontal)

# --- Inicializa figura ---
fig, ax = plt.subplots(figsize=(8, 8))

def atualizar(frame_idx):
    Ex_atual = valores_Ex[frame_idx]
    Ey_atual = valores_Ey[frame_idx]
    config_sim['E_externo'] = [Ex_atual, Ey_atual]

    Ex_total, Ey_total, V_potencial, magnitude_E, mascara_condutor, mascara_isolante_total = calcular_campos_e_potencial(X, Y, config_sim)
    visualizar_frame(ax, X, Y, Ex_total, Ey_total, V_potencial, magnitude_E, mascara_condutor, mascara_isolante_total, config_sim)

# ----- SUA FUNÇÃO DE SIMULAÇÃO (simplificada aqui) -----
def gerar_frame():
    fig, ax = plt.subplots(figsize=(5, 5))

    # Simula um campo girando para efeito de visualização
    t = time.time() % 10
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    Ex = np.cos(t) * X
    Ey = np.sin(t) * Y
    magnitude = np.sqrt(Ex**2 + Ey**2)

    ax.streamplot(X, Y, Ex, Ey, color=magnitude, cmap='viridis')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='jpeg', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def gerar_stream(config_sim, X, Y, num_frames=100):
    fig, ax = plt.subplots(figsize=(13, 10))
    largura_condutor = config_sim['material_x_max'] - config_sim['material_x_min']

    # Movimento e campo externo
    x_inicio, x_fim = -5.0, 5.0
    posicoes_x_centro = np.concatenate([
        np.linspace(x_inicio, x_fim, num_frames // 2),
        np.linspace(x_fim, x_inicio, num_frames // 2)
    ])
    Ex = np.linspace(2.0, -2.0, num_frames)
    Ey = np.linspace(0.0, 2.0, num_frames)

    for frame in range(num_frames):
        # Atualiza parâmetros
        x_centro_atual = posicoes_x_centro[frame]
        config_sim['material_x_min'] = x_centro_atual - largura_condutor / 2
        config_sim['material_x_max'] = x_centro_atual + largura_condutor / 2
        config_sim['E_externo'] = np.array([Ex[frame], Ey[frame]])

        # Cálculo dos campos
        Ex_total, Ey_total, V_potencial, magnitude_E, mascara_condutor, mascara_isolante_total = \
            calcular_campos_e_potencial(X, Y, config_sim)

        # Visualiza frame
        ax.clear()
        visualizar_frame(ax, X, Y, Ex_total, Ey_total, V_potencial, magnitude_E,
                         mascara_condutor, mascara_isolante_total, config_sim)

        # Salva como JPEG em memória
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        frame_bytes = buf.read()
        buf.close()

        # Stream MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ----- ROTAS FLASK -----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    config_sim = obter_configuracoes_simulacao()
    x_pts = np.arange(config_sim['x_min'], config_sim['x_max'] + config_sim['resolucao'], config_sim['resolucao'])
    y_pts = np.arange(config_sim['y_min'], config_sim['y_max'] + config_sim['resolucao'], config_sim['resolucao'])
    X, Y = np.meshgrid(x_pts, y_pts)
    
    return Response(gerar_stream(config_sim, X, Y),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ----- INICIAR -----
if __name__ == '__main__':
    app.run(debug=True)
