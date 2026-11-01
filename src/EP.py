#Mateus Teixeira Pasoss - 25200393
#Alice Cotrim Sturm - 25204204

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Permite salvar figuras sem abrir interface gráfica
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Variaveis gerais
tol = 1.06e-6
itmax = 20

# ---------------------------------------------------------
# Sistema (a)
def aF(x, y):  
    f1 = x**3 - 3*x*(y**2) - 1.
    f2 = 3*(x**2)*y - y**3
    return np.array([f1, f2], dtype=float)

def aJ(x, y):
    j11 = 3*(x**2) - 3*(y**2)
    j12 = -6*x*y
    j21 = 6*x*y
    j22 = 3*(x**2) - 3*(y**2)
    return np.array([[j11, j12], [j21, j22]], dtype=float)

raiz_a = np.array([
    [1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [-0.5,  np.sqrt(3)/2]
], dtype=float)

# ---------------------------------------------------------
# Sistema (b)
def bF(x, y):
    f1 = x**5 - 10*(x**3)*(y**2) + 5*x*(y**4) - 1.0
    f2 = 5*(x**4)*y - 10*(x**2)*(y**3) + (y**5)
    return np.array([f1, f2], dtype=float)

def bJ(x, y):
    j11 = 5*(x**4) - 30*(x**2)*(y**2) + 5*(y**4)
    j12 = -20*(x**3)*y + 20*x*(y**3)
    j21 = 20*(x**3)*y - 20*x*(y**3)
    j22 = 5*(x**4) - 30*(x**2)*(y**2) + 5*(y**4)
    return np.array([[j11, j12], [j21, j22]], dtype=float)

raiz_b = np.array([
    [1.0, 0.0],
    [np.cos(2*np.pi/5), np.sin(2*np.pi/5)],
    [np.cos(4*np.pi/5), np.sin(4*np.pi/5)],
    [np.cos(6*np.pi/5), np.sin(6*np.pi/5)],
    [np.cos(8*np.pi/5), np.sin(8*np.pi/5)]
], dtype=float)

# ---------------------------------------------------------
# Resolve J(X)*△(X) = -F(X) por Regra de Cramer (xi = det(Ai)/det(A)

#  [△x ] = det([-f1,j12]                 [△y ] = det([j11,-f1]
#               [-f2,j22]) /det(J)                    [j21,-f2])/det(J)

def resolver_equacao(f1, f2, j11, j12, j21, j22):
    det = j11*j22 - j12*j21
    delta_x = ((-f1)*j22 - (-f2)*j12) / det
    delta_y = (j11*(-f2) - j21*(-f1)) / det
    return delta_x, delta_y

# ---------------------------------------------------------
# Metodo de iteração compartilhada
def resolver_iteracao(F, get_J, x0, y0):
    x = float(x0)
    y = float(y0)
    it = 0
    
    with np.errstate(all="ignore"): #*Comando que não permite o terminal encher de avisos
        while it < itmax:
            F_val = F(x, y)
            if np.linalg.norm(F_val) < tol: # CRITÉRIO DE PARADA POR TOLERÂNCIA: Calcula a norma euclidiana(quantificação dos vetores) do residuo e verifica
                break

            f1, f2 = F_val[0], F_val[1] # Pega os valores de cada função
            
            jacobiana = get_J(x, y)
            j11, j12 = jacobiana[0, 0], jacobiana[0, 1]
            j21, j22 = jacobiana[1, 0], jacobiana[1, 1]
            
            delta_x, delta_y = resolver_equacao(f1, f2, j11, j12, j21, j22)
            
            #X(k+1) = X(k)+delta(X)
            x += delta_x
            y += delta_y
            it += 1
        
    return x, y, it

# ---------------------------------------------------------
# Método newton normal
def newton(F, J, x0, y0):
    return resolver_iteracao(F, J, x0, y0)

# ---------------------------------------------------------
# Método newton modificado
def newton_modificado(F, J, x0, y0):
    J_fixo = J(x0, y0)
    def J_estatico(x, y):
        return J_fixo
    return resolver_iteracao(F, J_estatico, x0, y0)

# ---------------------------------------------------------
# Método newton discreto
def newton_discreto(F, x0, y0, h=0.6):
    def J_discreto(x, y):
        F_atual = F(x, y)
        coluna_x = (F(x + h, y) - F_atual) / h  #(1/h)*[f(x+h,y)-f(x,y)]
        coluna_y = (F(x, y + h) - F_atual) / h  #(1/h)*[f(x,y+h)-f(x,y)]
        return np.column_stack((coluna_x, coluna_y)) #União da coluna_x e coluna_y para formar a matriz jacobiana
    
    return resolver_iteracao(F, J_discreto, x0, y0)

# ---------------------------------------------------------
# Identifica a qual raiz teórica o ponto final pertence 
def classificar_raiz(x, y, raizes, tol=tol):
    for r_idx, (xr, yr) in enumerate(raizes, start=1): #Percorre todas as raízes
        if np.hypot(x - xr, y - yr) < tol: #Faz o teorema de pitágoras e verifica se a distância entre o ponto final do método e a raiz é a menor que a tolerância.
            return r_idx
    return 0  # Significa que não convergiu para nenhuma raiz.

# ---------------------------------------------------------
# Gera os mapas C e M e salva o arquivo de imagem
def gerar_mapas(metodo_func, F, J, raizes, lim, N, titulo, nome_arquivo):
    coords = np.linspace(-lim, lim, N + 1)
    C = np.zeros((N + 1, N + 1), dtype=int)
    M = np.zeros((N + 1, N + 1), dtype=int)
    
    print(f"Calculando {titulo} (malha {N+1}x{N+1})...")
    
    for i in range(N + 1):
        y0 = coords[i] # Deslocamento vertical, equivalentes a linhas na matriz
        for j in range(N + 1):
            x0 = coords[j] # Deslocamento horizontal, equivalentes a colunas na matriz
            
            # Chama o método escolhido
            if J is not None:
                x_fim, y_fim, it = metodo_func(F, J, x0, y0)
            else:
                x_fim, y_fim, it = metodo_func(F, x0, y0) # Newton discreto não pede jacobiana
                
            raiz_id = classificar_raiz(x_fim, y_fim, raizes) # Verifica pra qual raiz a coordenada convergiu
            C[i, j] = raiz_id # Grava no mapa C a raiz convergida
            M[i, j] = it # Grava no mapa M  a quantidade de iterações

    # Plotagem
    L = len(raizes)
    cores = ["black", "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"][:L + 1] # Cria a paleta de cores, com 0 = preto e tons vívidos graduais até L+1
    extent = [-lim, lim, -lim, lim]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5)) # Prepara uma janela com dois quadros, ax[0] à esquerda e a[x1] à direita.
    
    # Mapa C
    ax[0].imshow(C, origin="lower", extent=extent, cmap=ListedColormap(cores), vmin=0, vmax=L)
    ax[0].set_title(f"Mapa C - {titulo}")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    
    # Mapa M
    cmap_m = plt.get_cmap("viridis").copy()
    cmap_m.set_bad("black")
    im = ax[1].imshow(np.ma.masked_where(M == 0, M), origin="lower", extent=extent, cmap=cmap_m, vmin=1, vmax=itmax)
    ax[1].set_title(f"Mapa M - {titulo}")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    fig.colorbar(im, ax=ax[1], label="Iterações") 
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=130)
    plt.close(fig)
    print(f"Figura salva: {nome_arquivo}")

# ---------------------------------------------------------
# ---------------------------------------------------------

x_teste, y_teste = 1.2, 0.2 # Área na bacia de atração da raiz [1.0,0.0]
xn, yn, itn = newton(aF, aJ, x_teste, y_teste)
xm, ym, itm = newton_modificado(aF, aJ, x_teste, y_teste)
xd, yd, itd = newton_discreto(aF, x_teste, y_teste)

print(f"Teste Ponto Inicial: ({x_teste}, {y_teste})")
print(f"Newton Padrão:     x={xn:.4f}, y={yn:.4f} | Iterações: {itn}")
print(f"Newton Modificado: x={xm:.4f}, y={ym:.4f} | Iterações: {itm}")
print(f"Newton Discreto:   x={xd:.4f}, y={yd:.4f} | Iterações: {itd}\n")

#Resolução da grade
N_pontos = 500  

# Mapas do sistema(a)
lim_a = 2.0
gerar_mapas(newton, aF, aJ, raiz_a, lim_a, N_pontos, "Sistema (a) - Newton Padrão", "mapa_a_newton.png")
gerar_mapas(newton_modificado, aF, aJ, raiz_a, lim_a, N_pontos, "Sistema (a) - Newton Modificado", "mapa_a_modificado.png")
gerar_mapas(newton_discreto, aF, None, raiz_a, lim_a, N_pontos, "Sistema (a) - Newton Discreto", "mapa_a_discreto.png")

# Mapas do sistema(b)
lim_b = 2.0
gerar_mapas(newton, bF, bJ, raiz_b, lim_b, N_pontos, "Sistema (b) - Newton Padrão", "mapa_b_newton.png")
gerar_mapas(newton_modificado, bF, bJ, raiz_b, lim_b, N_pontos, "Sistema (b) - Newton Modificado", "mapa_b_modificado.png")
gerar_mapas(newton_discreto, bF, None, raiz_b, lim_b, N_pontos, "Sistema (b) - Newton Discreto", "mapa_b_discreto.png")
