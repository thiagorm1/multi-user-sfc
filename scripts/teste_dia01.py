from muar_sfc.core.net_v2 import Net2


def teste_dia_1():
    print("--- INICIANDO TESTE FUNCIONAL DIA 1 ---")
    try:
        net = Net2()

        # 1. Criar Rede (Verifica se nasce com is_active=True)
        print("1. Criando nó S1...")
        net.add_node("S1", "server_edge", cpu_capacity=100)
        node = net.graph.nodes["S1"]

        if node.get("is_active") is True:
            print("   [OK] Nó nasceu com is_active=True.")
        else:
            print("   [ERRO] Nó não tem a flag is_active ou é False.")
            return

        # 2. Crash (Verifica se morre e se a trava de segurança sumiu)
        print("2. Simulando Crash...")
        net.set_node_down("S1")

        if node["is_active"] is False:
            print("   [OK] Flag mudou para False.")
        else:
            print("   [ERRO] Flag continua True após crash.")

        if node["cpu_capacity"] == 0:
            print("   [OK] Capacidade zerada.")
        else:
            print("   [ERRO] Capacidade não zerou.")

        # 3. Teste de Bloqueio (Tenta alocar no morto)
        print("3. Testando Guarda de Alocação...")
        try:
            # Mock de objetos para não precisar importar tudo
            class MockSFC:
                id = "sfc_1"

            class MockVNF:
                id = "vnf_1"

                def get_cpu_request(self):
                    return 10

                def get_cache_request(self):
                    return 0

            # Tenta alocar no nó S1 (que está morto)
            net.allocate_microservice(MockSFC(), MockVNF(), "S1")
            print("   [ERRO] A função permitiu alocar em nó morto! (Deveria dar erro)")
        except ValueError as e:
            if "inativo" in str(e) or "Crash" in str(e):
                print(f"   [OK] Bloqueio funcionou! Erro capturado: {e}")
            else:
                print(f"   [ALERTA] Deu erro, mas mensagem diferente do esperado: {e}")

        # 4. Recovery
        print("4. Recuperando S1...")
        net.restore_node("S1")

        if node["is_active"] is True and node["cpu_capacity"] == 100:
            print("   [OK] Nó ressuscitou com 100 de capacidade.")
        else:
            print(
                f"Fail_recuperação. Active={node.get('is_active')}, CPU={node.get('cpu_capacity')}"
            )

    except Exception as e:
        print(f"--- FALHA GERAL NO TESTE: {e} ---")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    teste_dia_1()
