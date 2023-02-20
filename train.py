import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import tqdm


class SER(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        feats, lens = self.prepare_features(stage, batch.sig)
                
        encoder_outs = self.modules.encoder(feats.detach())

        encoder_outs = torch.sum(encoder_outs, dim=1) 
        encoder_outs = self.modules.label_lin(encoder_outs)
        pred_outputs = self.hparams.log_softmax(encoder_outs)
        predictions = {"pred_outputs": pred_outputs}

        return predictions, lens 


    def compute_objectives(self, predictions, batch, stage):
        # Compute NLL loss
        predictions, lens = predictions
        labels, label_lens = batch.labels_encoded

        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["pred_outputs"],
            targets=labels.squeeze(-1),
        )
        

        if stage != sb.Stage.TRAIN:
            # Monitor word error rate and character error rated at valid and test time.
            self.accuracy_metric.append(predictions["pred_outputs"].unsqueeze(1), labels, label_lens)
 
        return loss


    def prepare_features(self, stage, wavs):
        """Prepare features for computation on-the-fly
        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        wavs : tuple
            The input signals (tensor) and their lengths (tensor).
        """
        wavs, wav_lens = wavs
        wavs = self.hparams.resample(wavs)
        
        # Feature computation and normalization
        feats = self.hparams.compute_features(wavs)
        print(feats.size())
        print(feats)
        feats = self.modules.normalizer(feats, wav_lens)

        return feats, wav_lens   
    

    def on_stage_start(self, stage, epoch=None):
       # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.accuracy_metric = self.hparams.accuracy_computer()


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["ACC"] = self.accuracy_metric.summarize()


        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["ACC"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            ) 
            
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"]}, max_keys=["ACC"],
                num_to_keep=1
            )
            
            # early stopping
            if self.hparams.epoch_counter.should_stop(current=epoch, current_metric=stage_stats["ACC"]):
                self.hparams.epoch_counter.current = self.hparams.epoch_counter.limit


        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            with open(self.hparams.decode_text_file, "w") as fo:
                for utt_details in self.wer_metric.scores:
                    print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)

            
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    def run_inference(
            self,
            dataset, # Must be obtained from the dataio_function
            max_key, # We load the model with the highest ACC
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.checkpointer.recover_if_possible(max_key=max_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        total = 0
        with torch.no_grad():
            true_labels = []
            pred_labels = []
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 
                out = self.compute_forward(batch, stage=sb.Stage.TEST) 
                predictions, wav_lens = out
                
                # SER prediction
                topi, topk = predictions["pred_outputs"].topk(1)
                topk = topk.squeeze()

                labels, label_lens = batch.labels_encoded
                labels = labels.squeeze()

                topk = topk.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                
                try: 
                    for elem in labels:
                        true_labels.append(elem)

                    for elem in topk:
                        pred_labels.append(elem)
                except:
                    true_labels.append(labels)
                    pred_labels.append(topk)
        
        
        #true_labels = np.array(true_labels)
        #pred_labels = np.array(pred_labels)
        #np.save("predictions/jaka_true", true_labels)
        #np.save("predictions/jaka_pred", pred_labels)
        
        true = np.array(true_labels)
        pred = np.array(pred_labels)
        
        #np.save("output/true_topics.npy", true_topics)
        #np.save("output/pred_topics.npy", pred_topics)
        print('F1: ', f1_score(true, pred, average="micro"))
        print('Weighted F1: ', f1_score(true, pred, average="weighted"))
        print('UAR: ', balanced_accuracy_score(true, pred))



def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev/dev_ANRO.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev/dev_ANRO.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test/test_ANRO.json"), replacements={"data_root": data_folder})

    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = sb.dataio.dataio.read_audio(file_path)
        if len(sig.size()) == 2:
            sig = torch.mean(sig, dim=-1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("labels_encoded")
    def text_pipeline(label):
        labels_encoded = hparams["label_encoder"].encode_sequence_torch([label])
        yield labels_encoded

    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    hparams["label_encoder"].update_from_didataset(train_data, output_key="label")

    # save the encoder
    hparams["label_encoder"].save(hparams["label_encoder_file"])
    
    # load the encoder
    hparams["label_encoder"].load_if_possible(hparams["label_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "labels_encoded"])
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data




def main(device="cuda"):
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    
    # Trainer initialization
    ser_brain = SER(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

 
    # Dataset creation
    train_data, valid_data, test_data = data_prep("../data/MFCC/CV_9", hparams)
   
    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ###ser_brain.checkpointer.delete_checkpoints(num_to_keep=0)
        ser_brain.fit(
            ser_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    else:
        # evaluate
        print("Evaluating")
        ser_brain.run_inference(test_data, "ACC", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
