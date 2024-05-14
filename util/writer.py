import os


def write_experiment(args, config_str, best_config):

    WRITE_PATH = "results/{}/".format(args.dataset)
    os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
    
    f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write(best_config)
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()