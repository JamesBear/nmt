python -m nmt.nmt --out_dir=/tmp/nmt_model --inference_input_file=/tmp/nmt_data/my_infer_file.question --inference_output_file=/tmp/nmt_model/output_infer --vocab_prefix=/tmp/nmt_data/vocab --src=question --tgt=answer --ckpt=/tmp/nmt_model/best_bleu/translate.ckpt-99000
