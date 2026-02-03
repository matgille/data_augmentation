import os
import random
import sys
import shutil
import glob

def move_files(files, target_dir):
	for file in files:
		basename_no_extension = file.split(".")[0]
		for image_and_xml in glob.glob(f"{basename_no_extension}*"):
			shutil.move(image_and_xml, target_dir)

def main(corpus, proportions):

	# Puis on crée les corpus
	random.shuffle(corpus)
	test_corpus = []
	train_corpus = []
	dev_corpus = []

	proportions['test'] = round(proportions['test'] * len(corpus))
	proportions['dev'] = round(proportions['dev'] * len(corpus))
	proportions['train'] = round(proportions['train'] * len(corpus))

	test_corpus.extend(corpus[:proportions['test']])
	train_corpus.extend(corpus[proportions['test']:proportions['test'] + proportions['train']])
	dev_corpus.extend(corpus[proportions['test'] + proportions['train']:])

	try:
		os.makedirs(f"test")
	except FileExistsError:
		pass
	move_files(test_corpus, "test/")
	try:
		os.makedirs(f"train")
	except FileExistsError:
		pass
	move_files(train_corpus, "train/")
	try:
		os.makedirs(f"dev")
	except FileExistsError:
		pass
	move_files(dev_corpus, "dev/")


if __name__ == '__main__':
	input_corpus = sys.argv[1]
	all_files = glob.glob(input_corpus)
	proportions = {'train': .9, 'test': .1, 'dev': 0}
	main(all_files, proportions)