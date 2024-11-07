import numpy as np

class FunkSVD:
    def __init__(self, n_fatores, taxa_aprendizado, penalizacao, n_eps, avaliacoes_df):
        """
        Inicializa o modelo com os parâmetros fornecidos:
        n_fatores: número de fatores latentes (dimensão das matrizes fatoriais)
        taxa_aprendizado: taxa de aprendizado para ajuste dos parâmetros
        penalizacao: fator de regularização para evitar overfitting
        n_eps: número de iterações para o treinamento
        """
        self.n_fatores = n_fatores
        self.taxa_aprendizado = taxa_aprendizado
        self.penalizacao = penalizacao
        self.n_eps = n_eps
        
        """
        Prepara o ambiente de treinamento:
        - Embaralha os dados
        - Mapeia IDs de usuários e itens
        - Calcula a média global das avaliações
        - Inicializa as matrizes fatoriais P (usuários) e Q (itens)
        - Inicializa os vieses dos usuários e itens
        """
        self.avaliacoes = avaliacoes_df.sample(frac=1).reset_index(drop=True)                               # Embaralha as avaliações
        self.indices = self.indexar_ids(self.avaliacoes)                                                    # Mapeia IDs de usuários e itens para índices

                                                                                                            # Calcula a média global das avaliações
        self.media_global = np.mean(self.avaliacoes['Rating'])
        self.avaliacoes = self.avaliacoes.to_dict(orient='records')                                         # Converte as avaliações em dicionários

                                                                                                            # Inicializa as matrizes fatoriais e vieses
        self.num_usuarios = len(self.indices[0])
        self.num_itens = len(self.indices[1])

                                                                                                            # Matriz de fatores latentes dos usuários
        self.P = np.random.normal(0, 0.01, (self.num_usuarios, self.n_fatores))

                                                                                                            # Matriz de fatores latentes dos itens
        self.Q = np.random.normal(0, 0.01, (self.n_fatores, self.num_itens))

                                                                                                            # Vieses dos usuários e itens inicializados com zero
        self.bias_usuarios = np.zeros(self.num_usuarios)
        self.bias_itens = np.zeros(self.num_itens)
        
        

    def indexar_ids(self, avaliacoes_df):
        """
        Mapeia IDs de usuários e itens para índices inteiros.
        Isso facilita o uso de arrays e operações vetoriais nas próximas etapas.
        
        Retorna dois dicionários: 
        - user_to_index: mapeia usuários para índices inteiros
        - item_to_index: mapeia itens para índices inteiros
        """
        id_user = 0
        id_item = 0
        
        user_to_index = {}
        item_to_index = {}
        
        for _, linha in avaliacoes_df.iterrows():
            usuario = linha['UserId']
            item = linha['ItemId']
            
                                                                                                        # Se o usuário ainda não foi mapeado, atribui um novo índice
            if usuario not in user_to_index:
                user_to_index[usuario] = id_user
                id_user += 1
            
                                                                                                        # Se o item ainda não foi mapeado, atribui um novo índice
            if item not in item_to_index:
                item_to_index[item] = id_item
                id_item += 1

        return user_to_index, item_to_index

    
       

    def ajustar(self):
        """
        Função que treina o modelo ajustando as matrizes P, Q e os vieses ao longo de múltiplas iterações.
        - Usa gradiente descendente para minimizar o erro entre as previsões e os valores reais.
        """                                                                
        
        for _ in range(self.n_eps):                                                                         # Itera por cada ep de treinamento
            for registro in self.avaliacoes:                                                                # Itera sobre cada avaliação
                nota_real = registro['Rating']                                                              # Nota real dada pelo usuário
                usuario_idx = self.indices[0][registro["UserId"]]                                           # Índice do usuário
                item_idx = self.indices[1][registro["ItemId"]]                                              # Índice do item

                                                                                                            # Extraímos os fatores latentes do usuário e do item, e os vieses
                fator_usuario = self.P[usuario_idx, :]
                fator_item = self.Q[:, item_idx]
                vies_usu = self.bias_usuarios[usuario_idx]
                vies_item = self.bias_itens[item_idx]

                                                                                                            # Previsão da avaliação do usuário para o item, usando a média global
                predicao = self.media_global + np.dot(fator_usuario, fator_item) + vies_usu + vies_item

                                                                                                            # Cálculo do erro (diferença entre a nota real e a previsão)
                erro = nota_real - predicao

                                                                                                            # Atualização dos vieses dos usuários e itens com base no erro
                ajuste_vies_usuario = self.taxa_aprendizado * (erro - self.penalizacao * vies_usu)
                ajuste_vies_item = self.taxa_aprendizado * (erro - self.penalizacao * vies_item)
                self.bias_usuarios[usuario_idx] += ajuste_vies_usuario
                self.bias_itens[item_idx] += ajuste_vies_item

                                                                                                            # Atualização das matrizes fatoriais P (usuários) e Q (itens)
                ajuste_P = self.taxa_aprendizado * (erro * fator_item - self.penalizacao * fator_usuario)
                ajuste_Q = self.taxa_aprendizado * (erro * fator_usuario - self.penalizacao * fator_item)
                self.P[usuario_idx, :] += ajuste_P
                self.Q[:, item_idx] += ajuste_Q

    def estimar(self, usuario_idx, item_idx):
        """
        Estima a avaliação de um usuário para um item com base nos fatores latentes
        e nos vieses ajustados durante o treinamento.
        """
                                                                                                            # Fatores latentes e vieses do usuário e item
        fator_usuario = self.P[usuario_idx, :]
        fator_item = self.Q[:, item_idx]
        vies_usu = self.bias_usuarios[usuario_idx]
        vies_item = self.bias_itens[item_idx]

                                                                                                            # Previsão da nota utilizando a média global, fatores latentes e vieses
        predicao = self.media_global + np.dot(fator_usuario, fator_item) + vies_usu + vies_item
        return predicao

    def estimar_para_alvos(self, conjunto_alvos):
        """
        Gera previsões para um conjunto de pares de usuários e itens.
        - Se o usuário ou item não estiver no conjunto de treinamento, retorna a média global.
        - Limita as previsões no intervalo [0, 1] para manter consistência.
        """
        estimativas = []
        for id_alvo in conjunto_alvos['UserId:ItemId']:                                                     # Para cada par de usuário e item
            ids = id_alvo.split(':')                                                                        # Separar usuário e item
            usuario_idx = self.indices[0].get(ids[0])                                                       # Busca o índice do usuário
            item_idx = self.indices[1].get(ids[1])                                                          # Busca o índice do item

                                                                                                            # Se o usuário ou item não foi encontrado, usa a média global como previsão
            if usuario_idx is None or item_idx is None:
                estimativas.append([id_alvo, self.media_global])
            else:
                                                                                                            # Faz a previsão usando os fatores e vieses ajustados
                predicao = self.estimar(usuario_idx, item_idx)

                                                                                                            # Limita a predição para estar entre 0 e 1
                predicao = min(max(predicao, 0), 1)
                estimativas.append([id_alvo, predicao])

        return estimativas
