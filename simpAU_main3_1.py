# This script helps to run model with the optimum hyperparameter
from LG_model import AUEMOModel
import os


if __name__ == '__main__':

    save_dir = "/Storage/Projects/FER_GAN/code/simpleAU/ckpt_bp4d_try_41/"  # sys.argv[2]
    tb_dir = '/Storage/Projects/AU_Discriminator/results41/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # config1 = { "batch_size": 48,
    #             "save_path": save_dir,
    #             "beta": 0.5,
    #             "beta1": 0.2,
    #             'n_land':49,
    #             "step_size": 2,
    #             "lr_gamma": 0.5,
    #             "data_split_method": "subject",
    #             "extreme_transform": "extreme",
    #             "fusion_AU1": 0.8,
    #             "fusion_AU2": 0.2,
    #             "fusion_AU4": 0.8,
    #             "fusion_AU15": 0.2,
    #             "fusion_AU23": 0.2,
    #             "fusion_AU24": 0.2,
    #             "fusion_weight": [0.5,0.5,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.3],
    #             "gamma_AU1":0.5,
    #             "gamma_AU2":0,
    #             "gamma_AU4":2,
    #             "gamma_AU15":1,
    #             "gamma_AU23":0,
    #             "gamma_AU24":2,
    #             "layers": [1,1,2,2],
    #             "loss": "bce",
    #             "lr": 1e-3,
    #             "model_type": "localglobal",
    #             "n_aus": 12,
    #             "num_cuts": 2,
    #             "sample_dat": 0.5,
    #             "weight_decay": 1e-04,
    #             "weights": [4.45,5.65,5.27,2.18,1.89,1.70,1.77,2.23,6.48,2.92,5.97,6.97], #[6.48,5.97,6.97],#[4.45,5.65,5.27,2.18,1.89,1.70,1.77,2.23,6.48,2.92,5.97,6.97],
    #             'device':'cuda:0',
    #             'num_workers':12,
    #             'start_save':4
    #             }

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
    inpaint_model.run_train(36, resume_training=False, resume_from_iters=17, validate=True)
