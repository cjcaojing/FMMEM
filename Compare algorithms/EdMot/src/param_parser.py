"""Parameter parsing from the command line."""

import argparse

def parameter_parser(no):
    """
    A method to parse up command line parameters. By default it trains on the polbooks dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    ##网络
    network_name = "gplus"
    
    parser = argparse.ArgumentParser(description="Run EdMot.")
    
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="../input/realnetwork/"+network_name+".CSV",
	                help="Edge list csv.")

    parser.add_argument("--membership-path",
                        nargs="?",
                        default="../output/"+network_name+"_membership_"+str(no)+".json",
	                help="Cluster memberhip json.")

    parser.add_argument("--components",
                        type=int,
                        default=1,
	                help="Number of components. Default is 2.")

    parser.add_argument("--cutoff",
                        type=int,
                        default=50,
	                help="Minimal overlap cutoff. Default is 50.")


    return parser.parse_args()
