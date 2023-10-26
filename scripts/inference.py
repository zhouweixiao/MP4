import os
import torch
import rouge
import argparse
import pytorch_lightning as pl
from dataloader import get_dataloader
from transformers import BartConfig, BartTokenizer
from modeling.modeling_speaker_bart import SpeakerBartForConditionalGeneration


class Summarizer(pl.LightningModule):
    def __init__(self, args):
        super(Summarizer, self).__init__()
        self.args = args

        self.tokenizer = BartTokenizer.from_pretrained(args.model_path)
        self.config = BartConfig.from_pretrained(args.model_path)
        self.model = SpeakerBartForConditionalGeneration.from_pretrained(args.model_path, config=self.config)

        self.pad_token_id = self.tokenizer.pad_token_id

    def _prepare_attn_mask(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.pad_token_id] = 0 
        return input_ids, attention_mask  

    def _prepare_results(self, p, r, f, metric):
        return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

    def _evaluate(self, preds, tgts):
        for aggregator in ['Avg', 'Best']:
            print('Evaluation with {}'.format(aggregator))
            
            apply_avg = aggregator == 'Avg'
            apply_best = aggregator == 'Best'
           
            evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2,
                                    limit_length=True, length_limit=256,
                                    length_limit_type='words',
                                    apply_avg=apply_avg, apply_best=apply_best,
                                    alpha=0.5, weight_factor=1.2, stemming=True)

            scores = evaluator.get_scores(preds, tgts)

            for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                print(self._prepare_results(results['p'], results['r'], results['f'], metric))

    def _write_summs_to_files(self, summs, infer_path):
        save_path = os.path.join(
            infer_path,
            "max=%d_min=%d_beams=%d_lp=%.2f_nrns=%d" % (
                self.args.gen_max_length,
                self.args.gen_min_length,
                self.args.gen_beam_size,
                self.args.gen_length_penalty,
                self.args.gen_no_repeat_ngram_size
            )
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        for i, s in enumerate(summs):
            file_path = os.path.join(save_path, f'{i}.txt')
            with open(file_path, 'w') as f:
                f.write(s) 

    def _summ_generate(self, input_ids):
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
        return generated_summs

    def test_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False

        input_ids, tgts = batch
        generated_summs = self._summ_generate(input_ids)
        return {"generated_summs": generated_summs, "tgts": tgts}

    def test_epoch_end(self, outputs):
        generated_summs = [item for output in outputs for item in output['generated_summs']]
        tgts = [item for output in outputs for item in output['tgts']]
        self._evaluate(generated_summs, tgts)
        self._write_summs_to_files(generated_summs, self.args.infer_path)
        return None


def inference(args):
    if args.resume_ckpt:
        summarizer = Summarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        summarizer = Summarizer(args)

    trainer = pl.Trainer(
        gpus=1,
        logger=False,
        replace_sampler_ddp=False,
        checkpoint_callback=False,
        precision=32 if args.fp32 else 16,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate
    )

    test_dataloader = get_dataloader(args, summarizer.tokenizer, "test")

    trainer.test(summarizer, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with Our Fine-tuned SOTA Models")

    # general
    parser.add_argument("--mode", type=str, default="inference", help="Mode of operation, default is 'inference'")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to the checkpoint to resume from")

    # trainer
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=1, help="Progress bar refresh rate")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 precision")

    # dataset
    parser.add_argument("--dataset_path", type=str, default="../datasets/Downstream_Datasets", help="Path to the dataset")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["SAMSum", "DIALOGSUM", "TWEETSUMM"], help="Name of the dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--max_length_src", type=int, default=1024, help="Maximum length of source sequence")
    parser.add_argument("--max_length_tgt", type=int, default=1024, help="Maximum length of target sequence")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    # generate
    parser.add_argument("--gen_use_cache", action="store_true", help="Use cache for generation")
    parser.add_argument("--gen_max_length", type=int, default=100, help="Maximum length of generated sequence")
    parser.add_argument("--gen_min_length", type=int, default=5, help="Minimum length of generated sequence")
    parser.add_argument("--gen_beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--gen_length_penalty", type=float, default=1.0, help="Length penalty for generation")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=0, help="No repeat n-gram size")
    parser.add_argument("--gen_early_stopping", action="store_true", help="Use early stopping in generation")

    # save
    parser.add_argument("--infer_path", type=str, required=True, help="Path for inference")

    args = parser.parse_args()

    inference(args)
