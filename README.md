# Newton Fractal Visualizer

Implementação em Python voltada para a disciplina de cálculo numérico e à renderização gráfica de bacias de atração geradas pela convergência do Método de Newton-Raphson aplicado a polinômios no plano complexo.
O projeto mapeia a sensibilidade às condições iniciais na resolução de raízes não lineares, gerando representações visuais de fractais com base na iteração de convergência. Este projeto é um exercício-programa para a disciplina INE5202 da UFSC.

---

## Objetivos de Aprendizado

- Aplicar conceitos de cálculo numérico e análise matemática na resolução de raízes de equações não lineares.
- Praticar a computação científica e a vetorização de dados matriciais para otimização de processamento.
- Mapear comportamentos caóticos e convergência matemática em saídas gráficas bidimensionais.

---

## Conceitos Matemáticos e Técnicos Aplicados

- Método de Newton Padrão, Discreto e Modificado.
- Mapeamento de Cores por Bacia: Associação do índice da raiz convergida e da quantidade de iterações gastas para a definição de mapas de gradiente e cor.

---

## Tecnologias Utilizadas

- Linguagem: Python 3
- Bibliotecas: NumPy (computação vetorial e cálculo numérico) e Matplotlib (processamento e renderização de imagens)
- Interface: Terminal e visualizador gráfico de imagens

---

## Descrição do Algoritmo

- Amostragem de Malha: Geração de um grid bidimensional de números complexos com resolução configurável.
- Critério de Parada: Interrupção por convergência de tolerância mínima estabelecida ou por alcance do limite máximo de iterações.
- Exportação Gráfica: Geração de arquivos de imagem representando visualmente os limites das bacias de atração.

---

## Desenvolvimento e Implementações

- Projeto com foco na disciplina de Cálculo Numérico, com o núcleo de cálculo e renderização estruturado. As principais componentes incluem:

- Funções Polinopmais: Definição da função matemática avaliada e de sua respectiva Jacobiana.
- Metodos Iteradores: Funções feitas para asseguar os métodos de convergência.
- Gráficos: Sequência responsável pela interpolação de dados numéricos em matrizes de cores e exportação visual via Matplotlib.