import os
import torch
import rouge
import argparse
import pandas as pd
import pytorch_lightning as pl
from dataloader import get_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from transformers import get_linear_schedule_with_warmup
from transformers import BartConfig, BartTokenizer
from modeling.modeling_speaker_bart import SpeakerBartForConditionalGeneration
from loss.label_smoothing import label_smoothed_nll_loss


class Summarizer(pl.LightningModule):
    def __init__(self, args):
        super(Summarizer, self).__init__()
        self.args = args

        self.tokenizer = BartTokenizer.from_pretrained(args.model_path)
        self.config = BartConfig.from_pretrained(args.model_path)
        self.model = SpeakerBartForConditionalGeneration.from_pretrained(args.model_path, config=self.config)
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def _prepare_attn_mask(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.pad_token_id] = 0 
        return input_ids, attention_mask

    def _compute_rouge_batch(self, input_ids, gold_strs):
        input_ids, attention_mask = self._prepare_attn_mask(input_ids)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=self.args.gen_use_cache,
            max_length=self.args.gen_max_length,
            min_length=self.args.gen_min_length,
            num_beams=self.args.gen_beam_size,
            length_penalty=self.args.gen_length_penalty,
            no_repeat_ngram_size=self.args.gen_no_repeat_ngram_size,
            early_stopping=self.args.gen_early_stopping
        )
        generated_summs = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        generated_summs = [s.replace("<eor>", " ").replace("<eou>", " ") for s in generated_summs]
        
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2,
                                limit_length=True, length_limit=256,
                                length_limit_type='words',
                                apply_avg=True, apply_best=False,
                                alpha=0.5, weight_factor=1.2, stemming=True)

        rouge_scores = []
        for pred, ref in zip(generated_summs, gold_strs):
            scores = evaluator.get_scores([pred], [ref])
            rouge_score = []
            for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                rouge_score.extend([results['p'], results['r'], results['f']])
            rouge_scores.append(rouge_score)
        return rouge_scores

    def _compute_rouge_all(self, outputs):
        rouge_result_all = [r for b in outputs for r in b["rouge_scores"]]
        names = []
        for rouge in ["1", "2", "L"]:
            names.extend(["rouge-{}-r".format(rouge), "rouge-{}-p".format(rouge), "rouge-{}-f".format(rouge)])
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        rouge_results.loc["avg_score"] = avg

        avgr = (avg[2] + avg[5] + avg[8]) / 3
        metrics = avg
        print("\n")
        print("Validation Result at Step %d" % (self.global_step))
        print("Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f" % (metrics[0], metrics[1], metrics[2]))
        print("Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f" % (metrics[3], metrics[4], metrics[5]))
        print("Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f" % (metrics[6], metrics[7], metrics[8]))
        print("\n")
        return names, metrics, avgr

    def configure_optimizers(self):                                                                                                                          
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.max_steps,
        )   
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, output_ids):
        input_ids, attention_mask = self._prepare_attn_mask(input_ids)
        decoder_input_ids = output_ids[:, :-1].clone()
        decoder_input_ids[decoder_input_ids == self.eos_token_id] = self.pad_token_id
        decoder_attention_mask = decoder_input_ids != self.pad_token_id
        outputs = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                decoder_input_ids = decoder_input_ids,
                decoder_attention_mask = decoder_attention_mask,
                head_mask = None,
                decoder_head_mask = None,
                cross_attn_head_mask = None,
                encoder_outputs = None,
                past_key_values = None,
                inputs_embeds = None,
                decoder_inputs_embeds = None,
                labels = None,
                use_cache = self.config.use_cache,
                output_attentions = self.config.output_attentions,
                output_hidden_states = self.config.output_hidden_states,
                return_dict = self.config.return_dict
        )
        lm_logits = outputs[0]
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        return lm_logits

    def shared_step(self, input_ids, output_ids):
        lm_logits = self.forward(input_ids, output_ids)
        labels = output_ids[:, 1:].clone()
        
        if self.args.label_smoothing == 0:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(lprobs, labels, self.args.label_smoothing, ignore_index=self.pad_token_id)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, output_ids = batch
        loss = self.shared_step(input_ids, output_ids)
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        return loss

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False 

        input_ids, output_ids, tgt = batch
        loss = self.shared_step(input_ids, output_ids)
        rouge_scores = self._compute_rouge_batch(input_ids, tgt)
        return {"vloss": loss, "rouge_scores": rouge_scores}
 
    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("vloss", vloss, sync_dist=True if self.args.use_ddp else False)

        names, metrics, avgr = self._compute_rouge_all(outputs)
        metrics = [vloss] + metrics
        names = ["vloss"] + names
        logs = dict(zip(*[names, metrics]))
        self.log("avgr", avgr)
        return {"avg_val_loss": vloss, "avgr": avgr, "log": logs, "progress_bar": logs}


def train(args):
    if args.resume_ckpt:
        model = Summarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = Summarizer(args)

    if not os.path.exists(args.ckpt_save_path):
        os.makedirs(args.ckpt_save_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_save_path,
        save_top_k=args.save_top_k,
        filename="{step}-{vloss:.2f}-{avgr:.4f}",
        monitor="avgr",
        mode="max",
    )

    trainer = pl.Trainer(
        logger=False,
        gpus=args.gpus,
        max_steps=args.max_steps,
        checkpoint_callback=True,
        callbacks=checkpoint_callback,
        precision=32 if args.fp32 else 16,
        val_check_interval=args.val_check_interval,
        accelerator="ddp" if args.use_ddp else None,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_val_batches=0.0 if args.few_shot else 1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        replace_sampler_ddp=True if args.use_ddp else False,
        plugins=DDPPlugin(find_unused_parameters=False) if args.use_ddp else None
    )

    train_dataloader = get_dataloader(args, model.tokenizer, "train")
    valid_dataloader = get_dataloader(args, model.tokenizer, "val")

    trainer.fit(model, train_dataloader, valid_dataloader)
    
    if args.few_shot:
        trainer.save_checkpoint(os.path.join(args.ckpt_save_path, "few-shot.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training and Fine-tuning with Our Models")

    # general
    parser.add_argument("--mode", type=str, required=True, choices=["pre-training-dap", "pre-training-top", "fine-tuning"], 
                        help="Specifies the mode of operation, either pre-training or fine-tuning.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model or model directory.")
    parser.add_argument("--resume_ckpt", type=str, default=None, 
                        help="Path to the checkpoint to resume from.")

    # checkpoint
    parser.add_argument("--ckpt_save_path", type=str, required=True, 
                        help="Directory where the checkpoints will be saved.")
    parser.add_argument("--save_top_k", type=int, default=100, 
                        help="number of top checkpoints to be saved.")

    # trainer
    parser.add_argument("--gpus", type=int, default=4, 
                        help="Number of GPUs to use for training.")
    parser.add_argument("--fp32", action="store_true", 
                        help="Use FP32 precision for training, instead of mixed precision.")
    parser.add_argument("--use_ddp", action="store_true", 
                        help="Enable Distributed Data Parallelism.")
    parser.add_argument("--max_steps", type=int, default=1150, 
                        help="Maximum number of training steps.")
    parser.add_argument("--val_check_interval", type=float, default=0.50, 
                        help="Interval for validation checks.")
    parser.add_argument("--num_sanity_val_steps", type=int, default=3, 
                        help="umber of sanity validation steps before starting the training.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, 
                        help="Number of batches to accumulate gradients before performing an optimization step.")
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=1, 
                        help="Refresh rate of the progress bar.")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5, 
                        help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=100, 
                        help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="A value of 0 means no smoothing, and a value closer to 1 makes the targets smoother.")

    # dataset
    parser.add_argument("--dataset_path", type=str, default="../datasets", 
                        help="Path to the dataset directory.")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        choices=["DAP_0.20", "DAP_0.40", "LCM3DS.json", "Downstream_Datasets/SAMSum", "Downstream_Datasets/DIALOGSUM", "Downstream_Datasets/TWEETSUMM"], 
                        help="Name of the dataset to use for training.")
    parser.add_argument("--val_dataset_path", type=str, default="../datasets/Downstream_Datasets", 
                        help="Path to the validation dataset directory.")
    parser.add_argument("--val_dataset_name", type=str, default=None, 
                        help="Name of the validation dataset.")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of worker threads for data loading.")
    parser.add_argument("--max_length_src", type=int, default=1024, 
                        help="Maximum length of the source sequences.")
    parser.add_argument("--max_length_tgt", type=int, default=256, 
                        help="Maximum length of the target sequences.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training and validation.")

    # few-shot
    parser.add_argument("--few_shot", action="store_true", 
                        help="Enable few-shot learning.")
    parser.add_argument("--seed", type=int, default=3442, 
                        help="Random seed for initialization.")
    parser.add_argument("--num_sample", type=int, default=10, 
                        help="Number of samples to use in few-shot learning.")

    # generate
    parser.add_argument("--gen_use_cache", action="store_true", 
                        help="Enable caching for faster generation.")
    parser.add_argument("--gen_max_length", type=int, default=100, 
                        help="Maximum length of generated sequences.")
    parser.add_argument("--gen_min_length", type=int, default=5, 
                        help="Minimum length of generated sequences.")
    parser.add_argument("--gen_beam_size", type=int, default=5, 
                        help="Size of the beam for beam search.")
    parser.add_argument("--gen_length_penalty", type=float, default=1.0, 
                        help="Length penalty for beam search.")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=0, 
                        help="Size of the no-repeat n-gram for beam search.")
    parser.add_argument("--gen_early_stopping", action="store_true", 
                        help="Enable early stopping in beam search when all beams predict EOS token.") 

    args = parser.parse_args()

    train(args)
