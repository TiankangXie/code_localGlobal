# This script helps to run model with the optimum hyperparameter
from LG_model import AUEMOModel
import os


if __name__ == '__main__':

    save_dir = "/Storage/Projects/FER_GAN/code/simpleAU/ckpt_bp4d_try_42/"  # sys.argv[2]
    tb_dir = '/Storage/Projects/AU_Discriminator/results42/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    configs = {
        'device':'cuda:0',
        'num_workers':16,
        "batch_size": 64,
        "sample_dat": None,
        'save_path': save_dir,
        'tb_dir': tb_dir,
        'tb_comment': 'SmallNet_lr1e-3',

        'n_land':49,
        "n_aus": 12,
        "weights": [4.45,5.65,5.27,2.18,1.89,1.70,1.77,2.23,6.48,2.92,5.97,6.97],

        "lr": 1e-3,
        "step_size": 4,
        "lr_gamma": 0.2,
        'start_save':4
    }
    
    inpaint_model = AUEMOModel(configs=configs)
    inpaint_model.run_train(36, resume_training=False, resume_from_iters=17, validate=False)
