import os
import pickle
import shutil
import tempfile
import time  # ### ADIÇÃO ###: Importa o módulo 'time' para adicionar pausas


def salvar_variavel(variavel, nome_lista, pasta="variaveis_salvas", valor_unico=False):
    """
    Salva ou atualiza uma variável em uma lista dentro de um arquivo .pkl.
    Cria o diretório e o arquivo se não existirem.
    """
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")

    lista = []
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, "rb") as f:
            try:
                lista = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                # print(
                #     f"Aviso: Arquivo '{caminho_arquivo}' encontrado vazio ou "
                #     f"corrompido. Será sobrescrito."
                # )
                lista = []

    if valor_unico:
        if not lista:
            lista = [variavel]
    else:
        if variavel not in lista:
            lista.append(variavel)

    # ### MELHORIA: LÓGICA DE RETENTATIVA (RETRY) ###
    # Esta seção foi modificada para lidar com bloqueios de arquivo do OneDrive.

    max_tentativas = 5  # Tenta salvar até 5 vezes
    atraso_tentativa = 0.2  # Espera 200ms entre as tentativas

    for _tentativa in range(max_tentativas):
        try:
            with open(caminho_arquivo, "wb") as f:
                pickle.dump(lista, f)
            # Se o salvamento foi bem-sucedido, imprime a mensagem e sai do loop
            # print(
            #     f"Variável salva em '{caminho_arquivo}'. Total de itens na "
            #     f"lista: {len(lista)}."
            # )
            break  # Sai do loop de tentativas
        except (OSError, PermissionError):
            # Se ocorrer um erro de permissão ou I/O, espera e tenta novamente
            # print(
            #     f"Tentativa {_tentativa + 1}/{max_tentativas}: Falha ao salvar "
            #     f"'{caminho_arquivo}' devido a: {e}. Tentando novamente em "
            #     f"{atraso_tentativa}s..."
            # )
            time.sleep(atraso_tentativa)


def salvar_lista(
    lista_para_adicionar, nome_arquivo, pasta="variaveis_salvas", evitar_duplicatas=False
):
    """
    Atualiza uma lista em um arquivo .pkl, adicionando os itens de uma nova lista.

    Carrega a lista existente, adiciona os itens da 'lista_para_adicionar'
    e salva a lista combinada de volta no arquivo.

    Argumentos:
        lista_para_adicionar (list): A lista de novos itens para adicionar.
        nome_arquivo (str): O nome do arquivo .pkl (ex: "meus_links.pkl").
        pasta (str): O diretório onde o arquivo está/será salvo.
        evitar_duplicatas (bool): Se True (padrão), só adiciona itens que
                                  ainda não existem na lista salva.
                                  Se False, adiciona todos os itens.
    """

    # 1. Garante que o diretório (pasta) exista
    if not os.path.exists(pasta):
        try:
            os.makedirs(pasta)
        except OSError as e:
            print(f"Erro crítico ao criar o diretório '{pasta}': {e}")
            return

    # 2. Define o caminho completo do arquivo
    if not nome_arquivo.endswith(".pkl"):
        nome_arquivo = f"{nome_arquivo}.pkl"
    caminho_arquivo = os.path.join(pasta, nome_arquivo)

    # 3. Carrega a lista existente (exatamente como em 'salvar_variavel')
    lista_salva = []
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, "rb") as f:
            try:
                lista_salva = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                # print(
                #     f"Aviso: Arquivo '{caminho_arquivo}' encontrado vazio ou "
                #     f"corrompido. Será sobrescrito."
                # )
                lista_salva = []

    # Garante que 'lista_salva' é realmente uma lista
    if not isinstance(lista_salva, list):
        lista_salva = []

    # 4. Adiciona os novos itens da 'lista_para_adicionar'
    itens_adicionados_count = 0
    if evitar_duplicatas:
        # Lógica idêntica à sua 'salvar_variavel', mas em um loop
        for item in lista_para_adicionar:
            if item not in lista_salva:
                lista_salva.append(item)
                itens_adicionados_count += 1
        # print(f"{itens_adicionados_count} novos itens foram adicionados.")
    else:
        # Simplesmente concatena as listas (permitindo duplicatas)
        lista_salva.extend(lista_para_adicionar)
        itens_adicionados_count = len(lista_para_adicionar)
        # print(f"{itens_adicionados_count} itens foram adicionados (duplicatas permitidas).")

    # 5. Salva a lista combinada de volta no arquivo (lógica de retry)
    max_tentativas = 5
    atraso_tentativa = 0.2

    for _tentativa in range(max_tentativas):
        try:
            with open(caminho_arquivo, "wb") as f:
                pickle.dump(lista_salva, f)
            # print(
            #     f"Lista atualizada em '{caminho_arquivo}'. Total de itens: "
            #     f"{len(lista_salva)}."
            # )
            break
        except (OSError, PermissionError):
            # print(
            #     f"Tentativa {_tentativa + 1}/{max_tentativas}: Falha ao salvar "
            #     f"'{caminho_arquivo}' (Erro: {e})..."
            # )
            time.sleep(atraso_tentativa)
    else:
        print(f"Falha ao salvar o arquivo '{caminho_arquivo}' após {max_tentativas} tentativas.")


def carregar_lista(nome_lista, pasta="variaveis_salvas"):
    """
    Carrega de forma segura uma lista de um arquivo .pkl.
    (Esta função não precisou de alterações)
    """
    caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")

    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, "rb") as f:
            try:
                return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                print(
                    f"Aviso: Arquivo '{caminho_arquivo}' encontrado vazio ou "
                    f"corrompido. Retornando lista vazia."
                )
                return []
    else:
        return []





def salvar_duas_variaveis_seguramente(
    variavel1, nome_lista1, variavel2, nome_lista2, pasta="variaveis_salvas", valor_unico=False
):
    """
    Salva duas variáveis em arquivos separados, mas somente se ambas puderem ser
    salvas com sucesso.
    Usa arquivos temporários e substitui os arquivos reais apenas após sucesso em ambos.
    """
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    def preparar_lista(variavel, nome_lista):
        caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")
        lista = []

        if os.path.exists(caminho_arquivo):
            with open(caminho_arquivo, "rb") as f:
                try:
                    lista = pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    lista = []

        if valor_unico:
            if not lista:
                lista = [variavel]
        else:
            if variavel not in lista:
                lista.append(variavel)

        return lista

    def salvar_temporario(lista, nome_lista):
        temp_fd, temp_path = tempfile.mkstemp(dir=pasta, suffix=".pkl")
        os.close(temp_fd)
        for _tentativa in range(5):
            try:
                with open(temp_path, "wb") as f:
                    pickle.dump(lista, f)
                return temp_path
            except (OSError, PermissionError):
                time.sleep(0.2)
        os.remove(temp_path)
        return None

    # Preparar listas
    lista1 = preparar_lista(variavel1, nome_lista1)
    lista2 = preparar_lista(variavel2, nome_lista2)

    # Tentar salvar temporariamente
    temp1 = salvar_temporario(lista1, nome_lista1)
    temp2 = salvar_temporario(lista2, nome_lista2)

    if temp1 and temp2:
        try:
            shutil.move(temp1, os.path.join(pasta, f"{nome_lista1}.pkl"))
            shutil.move(temp2, os.path.join(pasta, f"{nome_lista2}.pkl"))
            return True
        except Exception:
            # Rollback se algo der errado
            if os.path.exists(temp1):
                os.remove(temp1)
            if os.path.exists(temp2):
                os.remove(temp2)
            return False
    else:
        if temp1 and os.path.exists(temp1):
            os.remove(temp1)
        if temp2 and os.path.exists(temp2):
            os.remove(temp2)
        return False


def dividir_em_n_grupos(lista_completa, numero_de_grupos):
    """
    Divide uma lista em 'n' (numero_de_grupos) sublistas.

    Se o tamanho da lista não for perfeitamente divisível por 'n',
    os primeiros (n-1) grupos terão o tamanho base (divisão inteira),
    e o último grupo conterá todos os itens restantes.

    Argumentos:
        lista_completa (list): A lista de entrada a ser dividida.
        numero_de_grupos (int): O número de sublistas desejado.

    Retorna:
        list: Uma lista contendo as sublistas (grupos).
    """

    # --- Tratamento de casos especiais ---
    if not lista_completa:
        # Se a lista estiver vazia, retorna uma lista de N listas vazias
        return [[] for _ in range(numero_de_grupos)]

    if numero_de_grupos <= 0:
        # Não é possível dividir em 0 ou menos grupos
        raise ValueError("O número de grupos deve ser maior que zero.")

    # 1. Calcula o tamanho base de cada grupo (exceto o último)
    # Usamos divisão inteira (//)
    tamanho_total = len(lista_completa)
    tamanho_base = tamanho_total // numero_de_grupos

    # 2. Cria a lista de resultados e o ponteiro de índice
    resultado = []
    indice_atual = 0

    # 3. Cria os primeiros (n-1) grupos
    # Este loop rodará (numero_de_grupos - 1) vezes
    for _ in range(numero_de_grupos - 1):
        # Define o início e o fim de cada fatia (slice)
        inicio = indice_atual
        fim = indice_atual + tamanho_base

        # Adiciona a fatia à lista de resultados
        resultado.append(lista_completa[inicio:fim])

        # Atualiza o ponteiro para a próxima fatia
        indice_atual = fim

    # 4. Adiciona o último grupo
    # O último grupo pega tudo o que sobrou, do índice atual até o final da lista
    # Isso automaticamente inclui o "resto"
    resultado.append(lista_completa[indice_atual:])

    return resultado
