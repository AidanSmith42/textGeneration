
# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
!pip install -q gpt-2-simple
import gpt_2_simple as gpt2
from datetime import datetime
from google.colab import files

"""## GPU

check CPU. CoLab uses some beefy hardware
"""

!nvidia-smi

"""## Downloading GPT-2


* `124M` (default): the "small" model, 500MB on disk.
* `355M`: the "medium" model, 1.5GB on disk.
* `774M`: doesnt work
* `1558M`: really doesnt work

Larger models have more knowledge, but take longer to finetune and longer to generate text. 
"""

gpt2.download_gpt2(model_name="355M")

"""## Mounting Google Drive

VM drive mounting to get a text file loaded for i/o
"""

gpt2.mount_gdrive()

"""## Uploading txt file
Upload **any smaller text file**  (<10 MB) and update the file name in the cell below, then run the cell.
"""

file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	data = requests.get(url)
	
	with open(file_name, 'w') as f:
		f.write(data.text)

gpt2.copy_file_from_gdrive(file_name)

"""## training ... may take a while.  TensorFlow magic, saving checkpoints, be sure to save if using coLab so its not a bunch of wasted time..

*  **`restore_from`**: Set to `fresh` to start training from the base GPT-2, or set to `latest` to restart training from an existing checkpoint.
* **`sample_every`**: Number of steps to print example output
* **`print_every`**: Number of steps to print training progress.
* **`learning_rate`**:  Learning rate for the training. (default `1e-4`, can lower to `1e-5` if you have <1MB input data)
*  **`run_name`**: subfolder within `checkpoint` to save the model. This is useful if you want to work with multiple models (will also need to specify  `run_name` when loading the model)
* **`overwrite`**: Set to `True` if you want to continue finetuning an existing model (w/ `restore_from='latest'`) without creating duplicate copies.
"""

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='355M',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )

"""After the model is trained, you can copy the checkpoint folder to Drive.

do everything with Drive, its messy otherwise

"""

gpt2.copy_checkpoint_to_gdrive(run_name='run1')


"""## Generate Text From The Trained Model

After you've trained the model, `generate` generates a single text from the loaded model.
"""

gpt2.generate(sess, run_name='run1')
"""
You can generate multiple texts at a time by specifing `nsamples`.
 Unique to GPT-2, you can pass a `batch_size` to generate multiple samples in parallel, giving a massive speedup (in Colaboratory, set a maximum of 20 for `batch_size`).

*  **`length`**: Number of tokens to generate (default 1023, the maximum)
* **`temperature`**: The higher the temperature, the crazier the text (default 0.7, recommended to keep between 0.7 and 1.0)
* **`top_k`**: Limits the generated guesses to the top *k* guesses (default 0 which disables the behavior; if the generated output is super crazy, you may want to set `top_k=40`)
* **`top_p`**: Nucleus sampling: limits the generated guesses to a cumulative probability. (gets good results on a dataset with `top_p=0.9`)
* **`truncate`**: Truncates the input text until a given sequence, excluding that sequence (e.g. if `truncate='<|endoftext|>'`, the returned text will include everything before the first `<|endoftext|>`). It may be useful to combine this with a smaller `length` if the input texts are short.
*  **`include_prefix`**: If using `truncate` and `include_prefix=False`, the specified `prefix` will not be included in the returned text.
"""

gpt2.generate(sess,
              length=250,
              temperature=0.7,
              prefix="LORD",
              nsamples=5,
              batch_size=5
              )

"""For bulk generation, you can generate a large amount of text to a file and sort out the samples locally on your computer. The next cell will generate a generated text file with a unique timestamp.

You can rerun the cells as many times as you want for even more generated texts!
"""

gen_file = 'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())

gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=500,
                      temperature=0.7,
                      nsamples=100,
                      batch_size=20
                      )

# may have to run twice to get file to download
files.download(gen_file)


"""# Etcetera

If the notebook has errors (e.g. GPU Sync Fail), force-kill the Colaboratory virtual machine and restart it with the command below:
"""

!kill -9 -1
