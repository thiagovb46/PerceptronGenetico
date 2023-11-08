from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
atributos = iris.data
classes = iris.target

atributos_do_conjunto = atributos[classes != 0]
classes_conjunto = classes[classes != 0]    
classes_conjunto = np.where(classes_conjunto == 1, 1, -1)

#População inicial
def inicializar_populacao(tamanho_populacao, dimensao_individuo):
    populacao = []
    for j in range(tamanho_populacao):
        individuo = np.random.uniform(-1, 1, dimensao_individuo)
        populacao.append(individuo)
    return populacao

# Mutação do filho
def mutacao(individuo, taxa_mutacao):
    #Verifica numero aleatório gerado e compara com a taxa de mutação para mutar o filho
    for i in range(len(individuo)):
        if np.random.uniform(-1, 1) < taxa_mutacao: 
            individuo[i] = np.random.uniform(-1, 1)
    return individuo

# Para cada individuo calcula a aptidao EMQ
def aptidao(individuo):
    pesos = individuo[:-1]
    viés = individuo[-1]
    erro_quadrado_total = 0
    for i in range(atributos_treinamento.shape[0]):
        entrada = atributos_treinamento[i]
        alvo = classes_treinamento[i]
        soma_ponderada = np.dot(entrada, pesos) + viés
        previsao = 1 if soma_ponderada >= 0 else 0
        erro = alvo - previsao
        erro_quadrado_total += erro ** 2
    emq = erro_quadrado_total / atributos_treinamento.shape[0]
    return emq

# Cruzamento dos pais selecionados
def crossover(pai1, pai2):
    ponto_corte = np.random.randint(0, len(pai1))
    #Une pais de acordo com o ponto de corte definido
    filho = np.concatenate((pai1[:ponto_corte], pai2[ponto_corte:]))
    return filho

for cont in range(3):   
    if cont == 0:
        proporcao_conn_teste = 0.1
    if cont == 1:
        proporcao_conn_teste = 0.3
    if cont == 2:
        proporcao_conn_teste = 0.5
    # Divisão entre treinamento e teste
    atributos_treinamento, atributos_teste, classes_treinamento, classes_teste = train_test_split(atributos_do_conjunto, classes_conjunto, test_size=proporcao_conn_teste, random_state=42)
    tamanho_conjunto = atributos_teste.shape[0]
    #Inicialização pesos/bias
    np.random.seed(0)
    pesos = np.random.rand(atributos_treinamento.shape[1])
    viés = np.random.rand()

    #Genético
    tamanho_populacao = 100
    dimensao_individuo = atributos_treinamento.shape[1] + 1  #Adiciona o viés na ultima posição
    taxa_mutacao = 0.2
    num_geracoes = 200
    populacao = inicializar_populacao(tamanho_populacao, dimensao_individuo)

    for geracao in range(num_geracoes):
        aptidoes = []
        for individuo in populacao:
            apt = aptidao(individuo)
            aptidoes.append(apt);
        melhor_individuo = populacao[np.argmin(aptidoes)]
        # populacao que contem apenas o melhor indivíduo, será a próxima populacao
        populacao_selecionada = [melhor_individuo]
        #Gera filhos 
        while len(populacao_selecionada) < tamanho_populacao:
            #seleciona dois indices da populacao para serem os pais
            indices = np.random.choice(len(populacao), size=2, replace=False)
            pai1 = populacao[indices[0]]
            pai2 = populacao[indices[1]]
            filho = crossover(pai1, pai2)
            filho = mutacao(filho, taxa_mutacao)
            populacao_selecionada.append(filho)
        populacao = populacao_selecionada

    melhor_pesos = melhor_individuo[:-1]
    melhor_bias = melhor_individuo[-1]

    pesos = melhor_pesos;
    viés = melhor_bias;

    # Execução de testes do classificador
    corretos = 0
    for i in range(tamanho_conjunto): #Tamanho conjunto de testes
        entrada = atributos_teste[i]
        objetivo = classes_teste[i]
        soma_ponderada = np.dot(entrada, pesos) + viés
        ativação = np.sign(soma_ponderada)
        if ativação == objetivo:
            corretos += 1

           
    acuracia = (corretos / tamanho_conjunto) * 100 #Quantidade de acertos/tamanho do conjunto

    print("Teste "+str(cont+1))
    print("     Tamanho conjunto de teste:", proporcao_conn_teste)
    print("     Acurácia obtida no conjunto de teste %:", acuracia)