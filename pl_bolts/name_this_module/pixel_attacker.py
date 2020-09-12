class PixelAttacker(LightningModule):

    def __init__(self, hparams):
        # init superclass
        super(PixelAttacker, self).__init__(hparams)
        self.batch_size = hparams.batch_size

        # transfer key pretrained info
        self.dataset_name, self.extraction_model, self.original_loss_name = self.__load_pretrained()

    def __load_pretrained(self):
        self.finetuned_model = Classifiers.load_from_metrics(
            weights_path=self.hparams.pretrained_weights_path,
            tags_csv=self.hparams.pretrained_tags_csv,
            on_gpu=self.hparams.on_gpu,
            # map_location=map_location
        )

        self.finetuned_model.freeze()

        return self.finetuned_model.hparams.dataset_name, self.finetuned_model.hparams.extraction_model, self.finetuned_model.hparams.loss_name

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        # extract features
        is_success, best_solution, best_score, result = one_pixel_attacker.one_pixel_attack(
            model=self.finetuned_model,
            imgs=x[0],
            true_labels=x[1],
            class_names=self.class_names
        )

        return is_success

    def training_step(self, data_batch, batch_nb):
        """
        Called inside the training loop
        :param data_batch:
        :return:
        """
        # data_batch = source lang for memory saving
        logits = self.forward(data_batch)
        loss = self.loss(logits, data_batch[1])

        tqdm_dic = {}
        return loss, tqdm_dic

    def validation_step(self, data_batch, batch_nb):
        """
        Called inside the validation loop
        :param data_batch:
        :return:
        """
        # forward pass
        logits = self.forward(data_batch)
        loss = self.loss(logits, data_batch[1])

        # pick prediction
        preds = torch.topk(logits, dim=1, k=1)[1].view(-1)

        # track labels
        labels = data_batch[1]

        output = {
            'val_loss': loss.item(),
            'labels': labels,
            'preds': preds,
        }
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        all_labels = []
        all_preds = []
        losses = []
        for output in outputs:
            all_labels.extend(output['labels'].data.cpu().numpy())
            all_preds.extend(output['preds'].data.cpu().numpy())
            losses.append(output['val_loss'])

        val_acc = accuracy_score(all_labels, all_preds)

        # calculate average loss
        val_loss_mean = np.mean(losses)
        tqdm_dic = {
            'val_loss': val_loss_mean,
            'val_acc': val_acc
        }

        # end early
        if self.current_epoch >= 1 and val_acc < self.hparams.val_1_cutoff:
            os._exit(0)

        return tqdm_dic

    def loss(self, logits, labels):
        if self.on_gpu:
            labels = labels.cuda()

        return F.nll_loss(logits, labels)

    def update_tng_log_metrics(self, logs):
        return logs

    # ---------------------
    # MODEL SAVING
    # ---------------------
    def get_save_dict(self):
        checkpoint = {
            'state_dict': self.state_dict(),
        }

        return checkpoint

    def load_model_specific(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'])
        pass

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.choose_optimizer(self.hparams.optimizer_name, model_params, {'lr': self.hparams.learning_rate}, 'optimizer')
        self.optimizers = [optimizer]
        return self.optimizers
