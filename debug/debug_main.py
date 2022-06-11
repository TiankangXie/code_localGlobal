# This script helps to run model with the optimum hyperparameter
from LG_model import AUEMOModel
import os


if __name__ == '__main__':

    save_dir = "/Storage/Projects/FER_GAN/code/simpleAU/ckpt_bp4d_try_42/"  # sys.argv[2]
    tb_dir = '/Storage/Projects/AU_Discriminator/results44/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    configs = {
        'device':'cuda:0',
        'num_workers':8,
        "batch_size": 32,
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
    inpaint_model.load_data(sample_dat=None, validate=False)
    inpaint_model.train_debug()



    def train_debug(self):
        """
        train the network with images, without landmarks
        """
        self.model.train()
        loss_train_total = 0
        loss_train_dice = 0
        loss_train_softmax = 0
        loss_train_land = 0
        loss_train_refine = 0

        train_stats_all = collections.defaultdict(dict)

        dic_items = list(self.dataloaders_train.items())
        random.shuffle(dic_items)

        for task, task_set in enumerate(dic_items):
            # task_train_stats_per_AU_list = collections.defaultdict(dict)
            # task_train_statistics_list = []
            # train_stats_all[task_set[0]] = {}
            task_train_preds = {}
            task_train_labels = {}
            task_train_stats = collections.defaultdict(list)

            print(f"currently training task: {task_set[0]}")
            data_loadinger = tqdm(task_set[1])
            for batch_index, (img, label, land) in enumerate(data_loadinger):
                img = Variable(img).to(self.device, dtype=torch.float)
                label = Variable(label).long().to(self.device)
                land = Variable(land).float().to(self.device)

                # Convert
                self.optimizer.zero_grad()
                x_land, attention_map = self.model.forward_debug(img)

                loss_land = self.loss_mse(x_land, land)
                
                total_loss = loss_land 

                loss_train_total += total_loss.item()
                loss_train_land += loss_land.item()

                total_loss.backward()
                self.optimizer.step()

                if batch_index == 150:
                    n_img_show = 12
                    for n_au_plot in range(attention_map.shape[1]):
                        c_att = attention_map[n_img_show, n_au_plot, :, :].detach().cpu().numpy()
                        resize_att = cv2.resize(c_att, (176, 176))
                        heatmap = cv2.applyColorMap(np.uint8(255 * resize_att), cv2.COLORMAP_JET)
                        # if use_rgb:
                        #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        heatmap = np.float32(heatmap) / 255

                        v_img = self.unorm(img[n_img_show]).permute((1,2,0)).detach().cpu().numpy()
                        v_img = v_img[:, :, ::-1]
                        cam = heatmap + v_img
                        cam = cam / np.max(cam)
                        fused_img = np.uint8(255 * cam)
                        self.tb.add_image(f'AU{self.avail_label_list[n_au_plot]}/train', np.transpose(fused_img, (2, 0, 1)), self.iteration)

