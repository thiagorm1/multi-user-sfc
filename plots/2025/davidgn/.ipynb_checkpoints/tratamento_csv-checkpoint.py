import os
import numpy as np

def extrair_dicio_csv(path):
    # Abre o arquivo CSV no caminho fornecido
    with open(path, newline='') as csvfile:
        # Lê a primeira linha (cabeçalhos) e divide por vírgulas
        lista_headers = csvfile.readline().strip().split(",")
        
        # Cria um dicionário com cabeçalhos como chaves e listas vazias como valores
        dicio_csv = {i: [] for i in lista_headers}
        
        # Loop para ler cada linha do arquivo CSV
        while True:
            linha = csvfile.readline()
            
            # Se a linha estiver vazia (fim do arquivo), sai do loop
            if linha == "":
                break
                
            # Divide a linha por vírgulas para obter os dados
            lista_linha = linha.strip().split(",")
            
            # Adiciona os dados à lista correspondente no dicionário
            for i in range(len(lista_linha)):
                dicio_csv[lista_headers[i]].append(lista_linha[i])
    
    return dicio_csv

def combinar_csvs_em_dicio(diretorio):
    # Dicionário para armazenar os dados de todos os CSVs
    dicio_completo = None

    # Percorre todos os arquivos no diretório
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.csv'):
            caminho_arquivo = os.path.join(diretorio, arquivo)
            
            # Extrai os dados do CSV atual
            dicio_atual = extrair_dicio_csv(caminho_arquivo)
            
            # Se for o primeiro CSV, inicializa o dicionário completo
            if dicio_completo is None:
                dicio_completo = dicio_atual
            else:
                # Caso contrário, combina os dados no dicionário completo
                for chave, valores in dicio_atual.items():
                    dicio_completo[chave].extend(valores)
    
    return dicio_completo

# a= combinar_csvs_em_dicio("results/results_flows/Kuririn_PPO_s_50_p_4_a_1.0_c_0")
# print(len(a["acceptance_rate"]))

def expand_dicts(*dicts):
    # Encontra o maior tamanho das listas nos dicionários
    max_length = max(max(len(v) for v in d.values()) for d in dicts)

    # Expande as listas de cada dicionário até o tamanho máximo
    expanded_dicts = []
    for d in dicts:
        expanded_d = {}
        for key, values in d.items():
            if len(values) < max_length:
                # Repete os últimos elementos até o tamanho desejado
                expanded_d[key] = values + [values[-1]] * (max_length - len(values))
            else:
                expanded_d[key] = values
        expanded_dicts.append(expanded_d)
    
    return expanded_dicts

def indices_maiores_que(lista, limite):
    indices = []
    for i, valor in enumerate(lista):
        valor = float(valor)
        if valor > limite:
            indices.append(i)
    return indices

import numpy as np

import numpy as np

def substituir_outliers_por_bigode_6x(valores):
    # Garante que todos os valores sejam float
    for i, valor in enumerate(valores):
        valores[i] = float(valor)
    dados = np.array(valores)

    # Calcula Q1, Q3 e IQR
    Q1 = np.percentile(dados, 25)
    Q3 = np.percentile(dados, 75)
    IQR = Q3 - Q1

    # Limites do bigode (1.5 * IQR)
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Define o "mais alto do bigode"
    valores_no_intervalo = dados[dados <= limite_superior]
    if len(valores_no_intervalo) > 0:
        bigode_superior = valores_no_intervalo.max()
    else:
        bigode_superior = limite_superior

    limite_substituicao = 6 * bigode_superior

    dados_corrigidos = []
    for v in dados:
        if v > limite_substituicao:
            dados_corrigidos.append(limite_substituicao)
        else:
            dados_corrigidos.append(v)

    return dados_corrigidos




def percentual_outliers(valores):
    for i, valor in enumerate(valores):
        valores[i] = float(valor)
    dados = np.array(valores)
    Q1 = np.percentile(dados, 25)
    Q3 = np.percentile(dados, 75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR *6

    outliers = (dados < limite_inferior) | (dados > limite_superior)
    percentual = (np.sum(outliers) / len(dados)) * 100 if len(dados) > 0 else 0

    return percentual
