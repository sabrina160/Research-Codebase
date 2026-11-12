# server.py
import flwr as fl
import logging


# Enable INFO-level logging
logging.basicConfig(level=logging.INFO)
print(" Starting Flower server for federated training...")

def fit_config(server_round: int):
    return {"server_round": server_round}

def main():
    print(" Flower server running on localhost:8080")

    # Start server and get the final aggregated parameters
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        #fraction_evaluate=1.0,
        min_fit_clients=1,
        #min_evaluate_clients=1,
        min_available_clients=1,
        on_fit_config_fn=fit_config,   #send round number
    )

     
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

    print(" Server finished all rounds")

    # Save each backbone separately
    

if __name__ == "__main__":
    main()
