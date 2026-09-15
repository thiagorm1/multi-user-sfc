import os
import pickle
import time # ### ADIÇÃO ###: Importa o módulo 'time' para adicionar pausas

def salvar_variavel(variavel, nome_lista, pasta='variaveis_salvas', valor_unico=False):
    """
    Salva ou atualiza uma variável em uma lista dentro de um arquivo .pkl.
    Cria o diretório e o arquivo se não existirem.
    """
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")

    lista = []
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, 'rb') as f:
            try:
                lista = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                # print(f"Aviso: Arquivo '{caminho_arquivo}' encontrado vazio ou corrompido. Será sobrescrito.")
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
    atraso_tentativa = 0.2 # Espera 200ms entre as tentativas
    
    for tentativa in range(max_tentativas):
        try:
            with open(caminho_arquivo, 'wb') as f:
                pickle.dump(lista, f)
            # Se o salvamento foi bem-sucedido, imprime a mensagem e sai do loop
            # print(f"Variável salva em '{caminho_arquivo}'. Total de itens na lista: {len(lista)}.")
            break # Sai do loop de tentativas
        except (IOError, PermissionError) as e:
            # Se ocorrer um erro de permissão ou I/O, espera e tenta novamente
            # print(f"Tentativa {tentativa + 1}/{max_tentativas}: Falha ao salvar '{caminho_arquivo}' devido a: {e}. Tentando novamente em {atraso_tentativa}s...")
            time.sleep(atraso_tentativa)



def carregar_lista(nome_lista, pasta='variaveis_salvas'):
    """
    Carrega de forma segura uma lista de um arquivo .pkl.
    (Esta função não precisou de alterações)
    """
    caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")

    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, 'rb') as f:
            try:
                return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                print(f"Aviso: Arquivo '{caminho_arquivo}' encontrado vazio ou corrompido. Retornando lista vazia.")
                return []
    else:
        print("Caminho para as listas não existe!!!")
        return []




import tempfile
import shutil

def salvar_duas_variaveis_seguramente(variavel1, nome_lista1, variavel2, nome_lista2, pasta='variaveis_salvas', valor_unico=False):
    """
    Salva duas variáveis em arquivos separados, mas somente se ambas puderem ser salvas com sucesso.
    Usa arquivos temporários e substitui os arquivos reais apenas após sucesso em ambos.
    """
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    def preparar_lista(variavel, nome_lista):
        caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")
        lista = []

        if os.path.exists(caminho_arquivo):
            with open(caminho_arquivo, 'rb') as f:
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
        temp_fd, temp_path = tempfile.mkstemp(dir=pasta, suffix='.pkl')
        os.close(temp_fd)
        for tentativa in range(5):
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(lista, f)
                return temp_path
            except (IOError, PermissionError):
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
        except Exception as e:
            # Rollback se algo der errado
            if os.path.exists(temp1): os.remove(temp1)
            if os.path.exists(temp2): os.remove(temp2)
            return False
    else:
        if temp1 and os.path.exists(temp1): os.remove(temp1)
        if temp2 and os.path.exists(temp2): os.remove(temp2)
        return False
