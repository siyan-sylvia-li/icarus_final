# ICARUS
#### (Interruption-CApable Replies Using Seq2Seq)

This is the code for **When can I Speak? Predicting initiation points for spoken dialogue agents**.

## 1. Installations
```
pip install librosa torchaudio torch transformers pandas
```
* Install fairseq for [wav2vec 1.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec) model.
  * Then edit the trainer files accordingly to indicate where you put your wav2vec_large.pt file
* You would also need to use [Meticulous](https://ashwinparanjape.github.io/meticulous-ml/) if you want to train your own GMM / Heatmap / MSE-based models. 

## 2. Data
The Switchboard dataset is not included in this directory, and would need to be downloaded separately. 

To reproduce the training, eval, and test data used in this work, you would need our time-aligned version of the Switchboard Dialogue Act corpus ([SwDA](https://github.com/cgpotts/swda)) and the original Switchboard audios and transcripts (the .mrk files). We have code in this repository to align individual utterances from SwDA `.utt.csv` files to word timestamps in the Switchboard LDC release, and we are working on releasing this timestamped version.

Then, extract the train, validation, and test set data in the following manner.

```shell
python3 swda_data_split.py --part="train" --seq --size=200
python3 swda_data_split.py --part="val" --seq --size=20
python3 swda_data_split.py --part="test" --seq --size=20
```

Using the `--seq` flag processes the dialogues sequentially. Not using this flag would execute parallel processing and you can adjust the size of the threadpool in `swda_data_split.py`.

The code expects a `full_data` directory in the same level as the `swda_data_split.py` script with the following structure:

|- `full_data`

|---- Switchboard Conversation

|-------- audio `.wav` file

|-------- transcript `.mrk` file with word-level timestamps

|-------- time-aligned SwDA transcript `.utt.ts.csv` file

Extracting the data should resulting in three `.hdf5` files in a new `processed_data/` directory.

## 3. Run Training
You can run training on three different types of models:
1. Gaussian Mixture Model (`icarus_gmm_trainer.py`)
2. Heatmap (`icarus_heatmap_trainer.py`)
3. MSE-based Regression Model (`icarus_min_trainer.py`)

Running `icarus_gmm_trainer.py` directly should reproduce our `GMM-WGR` results on an A100 GPU.

## 4. Evaluation
`icarus_models_preds.py` evaluates trained models for their MAE-True and MAE-Pred values. `icarus_silent_preds.py` evaluates the silence baseline.

Assuming that the trained model is `experiments/[NUM]/[MODEL_NAME.pt]`, then model performance can be evaluated like this:

```shell
python3 icarus_models_preds.py --model_path=experiments/NUM/MODEL_NAME.pt --exp_num=NUM
```

If you want to see the actual predictions from the models, you can add `--add_pred` tag, and the prediction results should be stored under `experiments/NUM/predictions.csv`.