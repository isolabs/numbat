from routines import dino_train

if __name__ == "__main__":
    fp_repo         = "/home/prang/dev/numbat"
    fp_configs      = f"{fp_repo}/experiments"  
    dino_train(f"{fp_configs}/initial.yaml")