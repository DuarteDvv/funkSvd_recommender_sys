import sys
import pandas as pd
from FunkSVD import FunkSVD

def main():
    if len(sys.argv) != 3:
        print("Uso: python main.py <ratings.csv> <targets.csv>")
        sys.exit(1)


    ratings_csv_name = sys.argv[1]
    targets_csv_name = sys.argv[2]

    try:
        ratings = pd.read_csv(ratings_csv_name)
        targets = pd.read_csv(targets_csv_name)
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Erro: O arquivo está vazio.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Erro: Erro ao analisar o arquivo CSV.")
        sys.exit(1)

    ratings[['UserId', 'ItemId']] = ratings['UserId:ItemId'].str.split(':', expand=True)
    ratings['Rating'] = ratings['Rating'].astype(float)
    ratings = ratings[['UserId', 'ItemId'] + ['Rating'] + [col for col in ratings.columns if col not in ['UserId', 'ItemId', 'Rating']]]
    ratings = ratings.drop(columns=['UserId:ItemId'])
    ratings['Rating'] = ratings['Rating'] / 5.0                            #Normaliza entre 0 e 1 para evitar erros numéricos

    svd = FunkSVD(100, 0.05, 0.02, 20, ratings)

    svd.ajustar()

    previsoes = svd.estimar_para_alvos(targets)

    print("UserId:ItemId,Rating")


    for id, previsao in previsoes:
        denormalized_rating = min(max(previsao * 5, 1), 5) 
        
        print(f"{id},{denormalized_rating:.2f}")


if __name__ == "__main__":
    main()
